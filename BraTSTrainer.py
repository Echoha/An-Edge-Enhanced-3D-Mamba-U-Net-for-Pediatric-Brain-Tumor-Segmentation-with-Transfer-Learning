import csv
import numpy as np
from tqdm import tqdm
from light_training.utils.lr_scheduler import PolyLRScheduler
from light_training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from light_training.loss.deepsupervision import DeepSupervisionWrapper
from light_training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance, hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
import torch.nn.functional as F
import tensorboard
set_determinism(123)
import os
def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

def calculate_averages(results):
    averages = [np.nanmean([x[i] for x in results if x[i] is not None]) for i in range(3)]
    return averages
class BraTSTrainer(Trainer):
    def _build_loss(self):
        loss = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
        return loss

                 
    def __init__(self, model,final_save_path, resume_training, is_adult,  model_save_path, logdir, lr, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(final_save_path=final_save_path,
                         resume_training=resume_training,
                         is_adult=is_adult,
                        model_save_path=model_save_path,
                        env_type=env_type,
                        max_epochs=max_epochs,
                        batch_size=batch_size, 
                        device=device, 
                        val_every=val_every, 
                        num_gpus=num_gpus, 
                        logdir=logdir, 
                        master_ip=master_ip, 
                        master_port=master_port, 
                        training_script=training_script,
                        train_process=12,)
        self.roi_size = [128, 128, 128]
        self.augmentation = True
        self.window_infer = SlidingWindowInferer(roi_size=self.roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = self.augmentation
        self.lr = lr
        self.device = device

        self.model = model
        self.model_save_path = model_save_path

        self.patch_size = self.roi_size

        self.best_mean_dice = 0.0

        
        self.train_process = 2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.loss = self._build_loss()
        

    def training_step(self, batch):
        image, label = self.get_input(batch) # image size: [1, 4, 128, 128, 128], label size: [1, 128, 128, 128]
        if self.model is not None:
            pred = self.model(image) # pred: [1, 3, 128, 128, 128]
        pred = pred.float()
        expand_label = label.unsqueeze(1)  # 将 label 的形状从 [1, 128, 128, 128] 改为 [1, 1, 128, 128, 128]
        loss = self.loss(pred, expand_label) 
        return loss 
    
    def convert_labels(self, labels):
        ## WT, TC, and ET
        result = [(labels == 1) | (labels == 2) | (labels == 3), (labels == 2) | (labels == 3), labels == 3]
       
        return torch.cat(result, dim=1).float()

    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        '''
        """AI is creating summary for validation_step

        Returns:
            [type]: [description]
        """        '''
        image, label = self.get_input(batch)
       
        output = self.model(image)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs, model_save_path):
        dices = val_outputs
        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()
        print(f"dices is {tc, wt, et}")
        mean_dice = (tc + wt + et) / 3 
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")
    
        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        if (self.epoch + 1) % 50 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")
