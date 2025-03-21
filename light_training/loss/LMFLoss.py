import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class FocalLoss_3d(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1.0

    def forward(self, inputs, targets):
        num_classes = targets.size(1)
        assert len(self.alpha) == num_classes, \
            f'Length of weight tensor must match the number of classes. got {num_classes} expected {len(self.alpha)}'
        #
        # inputs: [B, C, D, H, W] - 假定输入经过sigmoid或softmax处理
        # targets: [B, C, D, H, W] - 应为 one-hot 编码格式
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 计算概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class FocalLoss(nn.Module):
    # FocalLoss 类旨在通过降低易分类样本的权重来关注困难分类样本，这常用于解决类不平衡问题。
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha # 类别权重, 影响损失函数对不同类别的重视程度
        self.gamma = gamma # 调节模型对困难样本的关注度
    
    def forward(self, output, target):
        num_classes = output.size(1)
        #  # 确保提供的alpha长度与类别数一致
        assert len(self.alpha) == num_classes, \
            f'Length of weight tensor must match the number of classes. got {num_classes} expected {len(self.alpha)}'
        # 计算交叉熵
        logp = F.cross_entropy(output, target, self.alpha)
        # # 通过交叉熵得到每个样本的概率
        p = torch.exp(-logp)
        # # 计算Focal Loss
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    # 是根据类别样本数计算得到的边距列表。边距用于调整某些类别的决策边界，使得少数类的决策边界更宽松。
    def __init__(self, cls_num_list,device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        print("LDAM weights:",weight)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.cpu()))
        #m_list = m_list * (max_m / np.max(m_list))
        m_list = (m_list * (max_m / torch.max(m_list))).to(device)
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.device = device
        assert s > 0 
        self.s = s #  控制logit缩放的因子
        self.weight = weight # # 类别权重

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m #  # 减去对应的边距
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
class LMFLoss(nn.Module):
        def __init__(self,cls_num_list,device,weight,alpha=0.2,beta=0.2, gamma=2, max_m=0.8, s=5,add_LDAM_weigth=False): 
            super().__init__()
            #  # 计算每个类的边距
            self.focal_loss = FocalLoss(weight, gamma)
            if add_LDAM_weigth:
                LDAM_weight = weight
            else:
                LDAM_weight = None
            print("LMF loss: alpha: ", alpha, " beta: ", beta, " gamma: ", gamma, " max_m: ", max_m, " s: ", s, " LDAM_weight: ", add_LDAM_weigth)
            self.ldam_loss = LDAMLoss(cls_num_list,device, max_m, weight=LDAM_weight, s=s)
            self.alpha= alpha # # Focal Loss 的权重
            self.beta = beta # # LDAM Loss 的权重

        def forward(self, output, target):
            focal_loss_output = self.focal_loss(output, target)
            ldam_loss_output = self.ldam_loss(output, target)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss 