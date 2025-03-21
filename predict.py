import csv
import pickle
import re
import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train, my_train_val_test_loader_from_split_json
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance, hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
set_determinism(123)
import os
from light_training.prediction import Predictor
def calculate_averages(results):
    averages = [np.nanmean([x[i] for x in results if x[i] is not None]) for i in range(3)]
    return averages

class BraTSTrainer(Trainer):
    def __init__(self, model_name, resume_training, final_save_path, model_save_path, is_adult, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(final_save_path,  resume_training, is_adult, model_save_path,  env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.patch_size = patch_size
        self.augmentation = False
        self.model_name = model_name
    
    def convert_labels(self, labels):
        result = [(labels == 1) | (labels == 3) | (labels == 2), (labels == 2) | (labels == 3), labels == 3]
        
        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)

        return image, label, properties 

    def define_model(self):
        import sys
       
         from nets import MambaEdge3DUNet
        self.model =  MambaEdge3DUNet.gget_mamba_edge_3d_unet(
                num_input_channels=4,
                deep_supervision=False
        )


        model.load_state_dict(new_sd, strict=False)
       
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")
        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        

        return model, predictor
    
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch) 
        ddim = False
        model, predictor = self.define_model()
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)
        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties)
        model_output = model_output.argmax(dim=0)[None]
        if is_post_process:
            model_output = predictor.postprocess(model_output)
        if is_save_images:
            save_model_output = predictor.predict_noncrop_probability(model_output, properties)
            predictor.save_to_nii(save_model_output, 
                                raw_spacing=[1,1,1],
                                case_name = properties['name'][0],
                                save_dir=save_path)
        
        model_output = self.convert_labels_dim0(model_output)
        
        label = label[0]
        c = 3
        dices = []
        haus_distances = []
        haus95_distances = []
        for i in range(0, c):
            output_i = model_output[i].cpu().numpy()

            
            unique_values, counts = np.unique(output_i, return_counts=True)
            label_i = label[i].cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)
            hd = hausdorff_distance(output_i, label_i)
            haus_distances.append(hd)
            hd95 = hausdorff_distance_95(output_i, label_i)
            haus95_distances.append(hd95)

        all_dices.append(dices)
        all_haus.append(haus_distances)
        all_haus95.append(haus95_distances)
        all_names.append(properties['name'][0])
        self.write_results_to_csv(csv_path, all_dices, all_haus, all_haus95, all_names)
        return 0


    def write_results_to_csv(self, filepath, all_dices, all_hausdorffs, all_hausdorff95s, all_names):
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Group', 'Name', 'WT Dice Score', 'TC Dice Score', 'ET Dice Score', 'Mean Dice', 
                            'WT Hausdorff', 'TC Hausdorff', 'ET Hausdorff', 'Mean Hausdorff',
                            'WT Hausdorff95', 'TC Hausdorff95', 'ET Hausdorff95', 'Mean Hausdorff95'])

            for idx in range(len(all_dices)):
                row = [idx + 1]

                mean_dice = np.nanmean(all_dices[idx])
                mean_haus = np.nanmean(all_hausdorffs[idx])
                mean_haus95 = np.nanmean(all_hausdorff95s[idx])
                 
                row.append(all_names[idx]) 
                row.extend(all_dices[idx] + [mean_dice])
                row.extend(all_hausdorffs[idx] + [mean_haus])
                row.extend(all_hausdorff95s[idx] + [mean_haus95])
                writer.writerow(row)
            overall_means = {
                'dice': calculate_averages(all_dices),
                'haus': calculate_averages(all_hausdorffs),
                'haus95': calculate_averages(all_hausdorff95s)
            }
            overall_mean_dice = np.nanmean(overall_means['dice'])
            overall_mean_haus = np.nanmean(overall_means['haus'])
            overall_mean_haus95 = np.nanmean(overall_means['haus95'])

            writer.writerow(['Average'] + [] + overall_means['dice'] + [overall_mean_dice] +
                            overall_means['haus'] + [overall_mean_haus] +
                            overall_means['haus95'] + [overall_mean_haus95])
            print(overall_means['dice'], [overall_mean_dice], 
                            overall_means['haus95'], [overall_mean_haus95])
        def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3) | (labels == 2), (labels == 2) | (labels == 3),  labels == 3]
        return torch.cat(result, dim=0).float()
    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd
    
import os

class TrainerConfig:
    def __init__(self):
        self.env = "pytorch"
        self.max_epoch = 10
        self.batch_size = 2
        self.val_every = 2
        self.num_gpus = 1
        self.device = "cuda:6"
        self.patch_size = [128, 128, 128]
        self.model_name = "MambaEdge3DUNet"
        self.select = 'adult'  # 'adult' or 'ped'
        self.is_save_images = False
        self.is_post_process = True
        self.model_select = 0  # Assuming model_select is an integer to index model names

        # File paths
        self.model_path = ''
        self.csv_path = os.path.splitext(self.model_path)[0] + '.csv'
        self.save_path = "prediction_results"
        self.final_save_path = ''  # This should be specified if resume_training is 'continue'
        self.logdir = self.generate_logdir()
        os.makedirs(self.save_path, exist_ok=True)

        # Dataset paths based on selection
        self.data_dir = ''
        self.splits_final_json = ''
        if self.select == 'adult':
            self.data_dir = 'data/nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans_3d_fullres'
            self.splits_final_json = '.splits_adult.json'
        else:
            self.data_dir = 'data/nnUNet_preprocessed/Dataset237_BraTS2023PED/nnUNetPlans_3d_fullres'
            self.splits_final_json = '.splits_ped.json'

    def generate_logdir(self):
        return f"/data/results/{self.model_name}_log"

def create_trainer(config: TrainerConfig):
    print(f'Model is {config.model_name}')
    
    trainer = BraTSTrainer(
        model_name=config.model_name,
        resume_training='none',  # Can be modified based on requirements
        final_save_path=config.final_save_path,
        model_save_path='',  # Specify the path if needed
        is_adult=config.select == 'adult',
        env_type=config.env,
        max_epochs=config.max_epoch,
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
    config = TrainerConfig()
    
    # Load dataset
    train_ds, val_ds, test_ds = train_val_test_loader_from_split_json(config.data_dir, config.splits_final_json)
    
    # Create trainer and run validation
    trainer = create_trainer(config)
    trainer.validation_single_gpu(test_ds)


if __name__ == "__main__":
    main()


