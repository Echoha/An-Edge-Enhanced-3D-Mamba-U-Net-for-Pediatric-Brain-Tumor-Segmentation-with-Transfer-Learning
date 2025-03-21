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
# 从这个地方了解monai的dice
from monai.metrics import DiceMetric
# # 可以修改的
# from simple_umamba.nets.UMambaEnc_3d import *
import tensorboard
set_determinism(123)
import os

# data_dir = "/home/ubuntu/h_workspace/data/nnUNet_preprocessed/Dataset237_BraTS2023PED/nnUNetPlans_3d_fullres"
# logdir = f"/home/ubuntu/h_workspace/SegMamba-umamba/logs/segmamba"
# model_save_path = '/home/ubuntu/h_workspace/SegMamba-umamba/logs/segmamba/model'
# ped_json = '/home/ubuntu/h_workspace/SegMamba-umamba/splits_ped.json'
# # augmentation = "nomirror"
# augmentation = True

# env = "pytorch"
# max_epoch = 500
# batch_size = 2
# val_every = 2
# num_gpus = 1
# device = "cuda:6"
# roi_size = [128, 128, 128]
# num_classes = 4
# lr = 2e-3
def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

def calculate_averages(results):
    # 计算每个区域的平均值，忽略NaN
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

        # from simple_umamba.nets import UMambaEnc_3d_with_newCombinedCurvature

        # from simple_umamba.nets import UMambaEnc_3d_with_newCombinedCurvature
        # self.model = UMambaEnc_3d_with_newCombinedCurvature.get_umamba_enc_3d(
        #         num_input_channels=4,
        #         deep_supervision=False
        # )
        # from eocformer_models.eoformer import EoFormer
        # self.model = EoFormer(in_channels=4, out_channels=4, drop_path=0.1)
        # print(self.model)

        # import simple_umamba.UMambaBot_3d
        # self.model = simple_umamba.UMambaBot_3d.get_umamba_model(
        #         num_input_channels=4,
        #         deep_supervision=False
        # )
        # from model_segmamba.segmamba import SegMamba
        # self.model = SegMamba(in_chans=4,
        #                 out_chans=4,
        #                 depths=[2,2,2,2],
        #                 feat_size=[48, 96, 192, 384])
        # from simple_umamba.nets.my_nnunet import get_network
        # model_name = ['ResidualEncoderUNet', 'PlainConvUNet']
        # self.model = get_network(model_name[1])
        self.model = model
        self.model_save_path = model_save_path

        self.patch_size = self.roi_size

        self.best_mean_dice = 0.0

        # 创建 DiceLoss 实例，忽略背景类别
        #self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        # self.ce = nn.CrossEntropyLoss() 
        # self.mse = nn.MSELoss()ß
        
        self.train_process = 2
        # self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        # self.scheduler =  PolyLRScheduler(self.optimizer, lr, max_epoch)
        
        self.scheduler_type = "poly"
        self.loss = self._build_loss()
        # self.cross = nn.CrossEntropyLoss()
        

    def training_step(self, batch):
        image, label = self.get_input(batch) # image size: [1, 4, 128, 128, 128], label size: [1, 128, 128, 128]
        

        if self.model is not None:
            
            pred = self.model(image) # pred: [1, 3, 128, 128, 128]
        # 如果deep supervision的话 pred = pred[0].float()  # Convert predictions to float32 before loss computation
        pred = pred.float()
        
        expand_label = label.unsqueeze(1)  # 将 label 的形状从 [1, 128, 128, 128] 改为 [1, 1, 128, 128, 128]

        loss = self.loss(pred, expand_label) 
        # loss = self.dice_loss(pred, label)

        

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
        # if self.epoch >= 54:
        #     torch.save(self.model.state_dict(), 
        #                os.path.join(model_save_path, 
        #                                 f"model_{mean_dice:.4f}_epoch_{self.epoch}.pt"))


        if (self.epoch + 1) % 50 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")
#     def test(self, test_ds, model_path, csv_path, model):
#         from light_training.prediction import Predictor
#         from monai.data import DataLoader
#         torch.cuda.empty_cache()
#         model.load_state_dict(torch.load(model_path), strict=False)
#         model.eval()  # 切换到评估模式

#         window_infer = SlidingWindowInferer(roi_size=self.patch_size,
#                                         sw_batch_size=2,
#                                         overlap=0.5,
#                                         progress=True,
#                                         mode="gaussian")

#         predictor = Predictor(window_infer=window_infer,
#                               mirror_axes=[0,1,2])
        
#         test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True)

#         # new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
#         # self.model.load_state_dict(model_path)
        
        
#         len_ = len(test_ds)
#         all_dices = []
#         all_haus = []
#         all_haus95 = []
#         all_name = []
        
        
#         for idx, batch in tqdm(enumerate(test_loader), total=len_):
#             if idx == len_:
#                 break
#             with torch.no_grad():
#                 test_outputs = self.test_step(batch, predictor, model)
#                 all_dices.append(test_outputs['dice'])
#                 all_haus.append(test_outputs['haus'])
#                 all_haus95.append(test_outputs['haus95'])
#                 all_name.append(test_outputs['name'])
        
#         self.on_test_epoch_end(csv_path, all_dices, all_haus, all_haus95, all_name)

        
#     def test_step(self, batch, predictor, model):
#         data = batch['data']
#         target = batch['seg']
#         properties = batch['properties']
#         # print(properties)
#         name = batch['properties']['name'][0]
#         data = torch.Tensor(data)
#         target = torch.Tensor(target)

#         if isinstance(target, list):
#             target = [i.to(self.device, non_blocking=True) for i in target]
#         else:
#             target = target.to(self.device, non_blocking=True)
#         # target to 
#         target = target.long() 
#         target = self.convert_labels(target)

#         print(target.shape)
#         print(data.shape)
#         model_output = predictor.maybe_mirror_and_predict(data, model, device=self.device) # [1, 4, x, y, z]
#         model_output = predictor.predict_raw_probability(model_output, 
#                                                          properties=properties)
#         model_output = model_output.argmax(dim=1)[:, None]
#         # 输出结果
#         unique_values, counts = model_output.unique(return_counts=True)
#         print("唯一值:", unique_values)
#         print("计数:", counts)
#         print(model_output.shape)
#         model_output = self.convert_labels(model_output) # [1, 3, x, y, z]
#         unique_values, counts = model_output.unique(return_counts=True)

#         print("唯一值:", unique_values)
#         print("计数:", counts)
#          # 计算唯一值和计数
        
        
#         model_output = model_output[0]
#         target = target[0]
        
#         dice_scores = []
#         haus_distances = []
#         haus95_distances = []
#         for i in range(model_output.shape[0]):  # Iterate over classes

#             pred = model_output[i, ...].cpu().numpy()
#             gt = target[i, ...].cpu().numpy()
            
#             dice_score= dice(gt, pred)
#             dice_scores.append(dice_score)

#             # Calculate Hausdorff Distance
#             hd = hausdorff_distance(pred, gt)
#             haus_distances.append(hd)
            
#             # Calculate 95% Hausdorff Distance
#             hd95 = hausdorff_distance_95(pred, gt)
#             haus95_distances.append(hd95)
        
#         print({'wt': dice_scores[0],
#                 'tc': dice_scores[1],
#                 'et': dice_scores[2],
#                 'hd wt': haus_distances[0],
#                 'hd tc': haus_distances[1],
#                 'hd et': haus_distances[2],
#                 'hd95 wt': haus95_distances[0],
#                 'hd95 tc': haus95_distances[1],
#                 'hd95 et': haus95_distances[2],
#                 'name': name
#                 })
        
        
#         return {'dice': dice_scores, 'haus': haus_distances, 'haus95': haus95_distances, 'name': name}
#     def on_test_epoch_end(self, save_csv_path, all_dices,  all_hausdorffs, all_hausdorff95s, all_names):
#         self.write_results_to_csv(save_csv_path, all_dices, all_hausdorffs, all_hausdorff95s, all_names)
#     def write_results_to_csv(self, save_csv_path, all_dices, all_hausdorffs, all_hausdorff95s, all_names):
#             with open(save_csv_path, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Group', 'name', 'WT Dice Score', 'TC Dice Score', 'ET Dice Score', 'Mean Dice', 
#                                 'WT Hausdorff', 'TC Hausdorff', 'ET Hausdorff', 'Mean Hausdorff',
#                                 'WT Hausdorff95', 'TC Hausdorff95', 'ET Hausdorff95', 'Mean Hausdorff95'])

#                 for idx in range(len(all_dices)):
#                     row = [idx + 1]
#                     mean_dice = np.nanmean(all_dices[idx])
#                     mean_haus = np.nanmean(all_hausdorffs[idx])
#                     mean_haus95 = np.nanmean(all_hausdorff95s[idx])
#                     row.append(all_names[idx])  # 正确使用 append 添加单个元素
#                     row.extend(all_dices[idx] + [mean_dice])
#                     row.extend(all_hausdorffs[idx] + [mean_haus])
#                     row.extend(all_hausdorff95s[idx] + [mean_haus95])

#                     writer.writerow(row)
                
#                 # 计算整体平均值
#                 overall_means = {
#                     'dice': calculate_averages(all_dices),
#                     'haus': calculate_averages(all_hausdorffs),
#                     'haus95': calculate_averages(all_hausdorff95s)
#                 }
#                 overall_mean_dice = np.nanmean(overall_means['dice'])
#                 overall_mean_haus = np.nanmean(overall_means['haus'])
#                 overall_mean_haus95 = np.nanmean(overall_means['haus95'])

#                 writer.writerow(['Average'] + [''] + overall_means['dice'] + [overall_mean_dice] +
#                                 overall_means['haus'] + [overall_mean_haus] +
#                                 overall_means['haus95'] + [overall_mean_haus95])
#                 print(f"mean dice: {overall_mean_dice}")
#                 print(f"mean haus95: {overall_mean_haus95}")
#     def filte_state_dict(self, sd):
#         if "module" in sd :
#             sd = sd["module"]
#         new_sd = {}
#         for k, v in sd.items():
#             k = str(k)
#             new_k = k[7:] if k.startswith("module") else k 
#             new_sd[new_k] = v 
#         del sd 
#         return new_sd
    


# # if __name__ == "__main__":

#     # trainer = BraTSTrainer(env_type=env,
#     #                         max_epochs=max_epoch,
#     #                         batch_size=batch_size,
#     #                         device=device,
#     #                         logdir=logdir,
#     #                         val_every=val_every,
#     #                         num_gpus=num_gpus,
#     #                         master_port=17759,
#     #                         training_script=__file__)
#     # # ped_data_dir = data_dir
#     # # ped_splits_json = '/home/ubuntu/h_workspace/SegMamba-umamba/splits_ped.json'
#     # # adult_data_dir = '/home/ubuntu/h_workspace/data/nnUNet_preprocessed/Dataset337_BraTS_Mixed/nnUNetPlans_3d_fullres'
#     # # train, val, test = my_get_mixed_train_val_test_loader_from_split_json(ped_data_dir, ped_splits_json, adult_data_dir)
#     # train_ds, val_ds, test_ds =  my_ped_train_val_test_loader_from_split_json(data_dir, ped_json)
    
#     # # train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)


#     # trainer.train(train_dataset=train_ds, val_dataset=val_ds)
#     # # model_path = '/home/ubuntu/h_workspace/SegMamba-umamba/logs/segmamba/segmamba_train500epoch_model/best_model_segmamba_train500epoch_0.7708.pt'
#     # # csv_path = '/home/ubuntu/h_workspace/SegMamba-umamba/logs/segmamba/csv/best_model_segmamba_train500epoch_0.7708.csv'
#     # # trainer.test(test_ds=test_ds, model_path=model_path, csv_path=csv_path)
