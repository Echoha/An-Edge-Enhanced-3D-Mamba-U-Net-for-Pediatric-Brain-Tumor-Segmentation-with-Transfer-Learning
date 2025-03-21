# An-Edge-Enhanced-3D-Mamba-U-Net-for-Pediatric-Brain-Tumor-Segmentation-with-Transfer-Learning

## Abstract
Pediatric gliomas are highly aggressive tumors with low survival rates, and their segmentation remains challenging due to distinct imaging characteristics and data scarcity. While deep learning models perform well in adult glioma segmentation, they struggle with pediatric gliomas, particularly in segmenting complex regions such as the tumor core (TC) and enhancing tumor (ET). This study proposes a solution using a 3D Mamba U-Net model combined with transfer learning to address these challenges. The model integrates U-Net's multi-scale feature extraction with Mamba's global dependency modeling, enhancing long-range feature representation. A cross-layer residual connection within Mamba block further improves segmentation accuracy. Additionally, an Edge Enhancement module is embedded in the skip connection layers of the U-Net structure to better capture local features in small pediatric tumor regions and refine boundary detection. A decoder fine-tuning strategy is introduced to adapt the model to pediatric data by adjusting the final feature reconstruction stage while preserving knowledge from adult datasets. Experimental results demonstrate that the proposed approach significantly outperforms previous methods in pediatric glioma segmentation, addressing both the issue of complex tumor morphology and limited dataset size. On the BraTS-PEDs 2023 dataset, the average Dice scores for the whole tumor (WT), tumor core (TC), and enhanced tumor (ET) regions are 0.8917, 0.8557, and 0.6365, respectively. The Hausdorff distances for these regions are 3.8277, 5.1439, and 3.5365, respectively. These results highlight that the proposed method surpasses existing approaches, particularly in segmenting fine-grained tumor sub-regions.

## Usage

### Installation Guide

#### Requirements
- Ubuntu 20.04
- CUDA 11.8

#### Steps

1. **Create a Virtual Environment**  
   First, create a new conda environment and activate it:  
   ```bash
   conda create -n umamba python=3.10 -y  
   conda activate umamba
   ```

2. **Install PyTorch 2.0.1**  
   Install the specified version of PyTorch and torchvision:  
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Mamba**  
   Install the Mamba package:  
   ```bash
   pip install causal-conv1d>=1.2.0  
   pip install mamba-ssm --no-cache-dir
   ```

4. **Install MONAI**  
   Install the MONAI package:  
   ```bash
   pip install monai
   ```

### Sanity Test

After installation, verify the setup by running the following commands in a Python environment:
```python
import torch
import mamba_ssm
```

### Preprocessing

To preprocess the dataset, run the following command:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

Once preprocessing is complete, you can train your model with the following command:
```bash
python train.py
```

### Inference

After training, use the following command to run inference on the testing set:
```bash
python predict.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
