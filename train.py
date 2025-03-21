import numpy as np
from light_training.utils.lr_scheduler import PolyLRScheduler
from light_training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from light_training.loss.deepsupervision import DeepSupervisionWrapper
from light_training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer, save_checkpoint
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
import tensorboard
set_determinism(123)
import os

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)


class BraTSTrainer(Trainer):
    def _build_loss(self):
        loss = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
        return loss
   
    def __init__(self, model_name, resume_training, final_save_path, model_save_path, is_adult, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(final_save_path,  resume_training, is_adult,  model_save_path, env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        
        from nets import MambaEdge3DUNet
        self.model =  MambaEdge3DUNet.gget_mamba_edge_3d_unet(
                num_input_channels=4,
                deep_supervision=False
        )
        self.patch_size = [128, 128, 128]
        self.best_mean_dice = 0.0
        self.train_process = 2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.loss = self._build_loss()
        
        self.best_save_path = ''

    def training_step(self, batch):
        image, label = self.get_input(batch) 
        
        pred = self.model(image)
        pred = pred.float()
        expand_label = label.unsqueeze(1)  

        loss = self.loss(pred, expand_label) 

        return loss 
    
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 2) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
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
        choice = "adult" if self.is_adult else "ped"

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            
            self.best_save_path = os.path.join(model_save_path, 
                                            f"{choice}_best_model_{mean_dice:.4f}.pt")
            save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, self.best_save_path, f"{choice}_best_model")
        self.final_save_path = os.path.join(model_save_path, 
                                        f"{choice}_final_model_{mean_dice:.4f}.pt")
        save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, self.final_save_path, f"{choice}_final_model")
        if self.epoch >= self.max_epochs - 5:
            tmp_save_path = os.path.join(model_save_path, 
                                        f"{choice}_model_epoch_{self.epoch}_{mean_dice:.4f}.pt")
            save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, tmp_save_path, f"{choice}_model_epoch_{self.epoch}")
            

        if (self.epoch + 1) % 25 == 0:
            tmp_save_path = os.path.join(model_save_path, f"{choice}_model_ep{self.epoch}_{mean_dice:.4f}.pt")
            save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, tmp_save_path, f"{choice}_model_ep{self.epoch}")

        print(f"mean_dice is {mean_dice}")


class Config:
    def __init__(self):
        self.augmentation = True
        self.roi_size = [128, 128, 128]
        self.env = "pytorch"
        self.batch_size = 2
        self.val_every = 1
        self.num_gpus = 1
        self.device = "cuda:2"
        self.num_classes = 4
        self.lr = 2e-3

        self.adult_max_epoch = 50
        self.ped_max_epoch = 150

        # Dataset directories and files
        self.adult_data_dir = "data/nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans_3d_fullres"
        self.adult_json = '.splits_adult.json'
        self.ped_data_dir = "data/nnUNet_preprocessed/Dataset237_BraTS2023PED/nnUNetPlans_3d_fullres"
        self.ped_json = '.splits_ped.json'

        # Model configurations
        self.module = {'adult': 0, 'ped': 1}
        self.model_names = 'MambaEdge3DUNet'
        self.model_select = 0  
        self.select = 'adult'
        self.resume_training = 'none'

        # Final save path configuration
        self.final_save_path = ""
        self.logdir = self.generate_logdir()
        self.model_save_path = self.generate_model_save_path()

        # Max epochs for each dataset
        self.max_epochs = [self.adult_max_epoch, self.ped_max_epoch]

    def generate_logdir(self):
        logdir = f"/data05/hwh/results/{self.select}s_results/{self.model_names[self.model_select]}_log"
        return logdir

    def generate_model_save_path(self):
        model_save_path = f"/data05/hwh/results/{self.select}s_results/{self.model_names[self.model_select]}_model"
        os.makedirs(model_save_path, exist_ok=True)
        return model_save_path

    def set_is_adult(self):
        return self.select == 'adult' and self.resume_training == 'none'


def configure_trainer(config):
    print(f'Model selected: {config.model_names[config.model_select]}')
    print(f'Model save path: {config.model_save_path}')
    
    if config.select == 'adult':
        is_adult = True
    else:
        is_adult = False

    if config.resume_training != 'none':
        is_adult = False
        if not config.final_save_path:
            raise ValueError("Error: final save path cannot be an empty string")

    trainer = BraTSTrainer(
        model_name=config.model_names[config.model_select],
        resume_training=config.resume_training,
        final_save_path=config.final_save_path,
        model_save_path=config.model_save_path,
        is_adult=is_adult,
        env_type=config.env,
        max_epochs=config.max_epochs[config.module[config.select]],
        batch_size=config.batch_size,
        device=config.device,
        logdir=config.logdir,
        val_every=config.val_every,
        num_gpus=config.num_gpus,
        master_port=17759,
        training_script=__file__
    )
    
    return trainer


def main():
    config = Config()

    # Setup dataset and trainer
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        config.adult_data_dir if config.select == 'adult' else config.ped_data_dir, 
        config.adult_json if config.select == 'adult' else config.ped_json
    )

    trainer = configure_trainer(config)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


if __name__ == "__main__":
    main()

