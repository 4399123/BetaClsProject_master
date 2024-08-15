# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy,AsymmetricLossSingleLabel,JsdCrossEntropy
import warnings

def get_lossfunc(args,train_dataset,accelerator):

    #计算每个类的图像数
    class_num = len(train_dataset.classes)
    cls_num_list = list()
    for i in range(class_num):
        n = 0
        for j in train_dataset.imgs:
            if (i == j[1]):
                n += 1
        cls_num_list.append(n)

    accelerator.print(cls_num_list)

    # loss选择
    criterion = nn.CrossEntropyLoss()
    if args.loss == 0:
        accelerator.print('启动交叉熵损失')
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    elif args.loss == 1:
        accelerator.print('启动标签平滑')
        # criterion = LabelSmoothingCELoss()  # 标签平滑损失
        criterion = LabelSmoothingCrossEntropy(0.1)
    elif args.loss == 2:
        accelerator.print('启动FocalLoss')
        criterion = FocalLoss()  # FocalLoss
    elif args.loss == 3:
        accelerator.print('启动PolyCELoss')
        criterion = PolyCE(class_number=class_num)  # PolyCELoss
    elif args.loss == 4:
        accelerator.print('启动PolyFocalLoss')
        criterion = PolyFocalLoss(class_number=class_num)  # PolyFocalLoss
    elif args.loss == 5:
        accelerator.print('启动AsymmetricLoss')
        criterion = AsymmetricLossSingleLabel()  # AsymmetricLoss
    elif args.loss == 6:
        accelerator.print('启动LabelSmooth_AsymmetricLoss')
        criterion = LabelSmooth_AsymmetricLoss()  # 标签平滑+非对称损失
    elif args.loss == 7:
        accelerator.print('启动JsdCrossEntropy')
        criterion = JsdCrossEntropy( num_splits=class_num)  # JsdCrossEntropy
    elif args.loss == 8:
        accelerator.print('启动JsdCrossEntropy+非对称损失')
        criterion = JSD_AsymmetricLoss( num_splits=class_num)  # JsdCrossEntropy+非对称损失
    else:
        accelerator.print('不支持的loss')

    return criterion

#标签平滑+非对称损失
class LabelSmooth_AsymmetricLoss(nn.Module):
    def __init__(self):
        super(LabelSmooth_AsymmetricLoss,self).__init__()
        self.ls=LabelSmoothingCrossEntropy(0.1)
        self.al=AsymmetricLossSingleLabel()

    def forward(self,output,target):
        lsloss=self.ls(output,target)
        aloss=self.al(output,target)
        return  lsloss+aloss

#JsdCrossEntropy+非对称损失
class JSD_AsymmetricLoss(nn.Module):
    def __init__(self,num_splits):
        super(JSD_AsymmetricLoss,self).__init__()
        self.jsd=JsdCrossEntropy(alpha=6,num_splits=num_splits)
        self.al=AsymmetricLossSingleLabel()

    def forward(self,output,target):
        jsdloss=self.jsd(output,target)
        aloss=self.al(output,target)
        return  0.5*jsdloss+aloss


#标签平滑损失
class LabelSmoothingCELoss(nn.Module):
    def __init__(self,e=0.1):
        super(LabelSmoothingCELoss,self).__init__()
        self.e=e

    def forward(self,output,target):
        num_classes=output.size()[-1]
        log_preds=F.log_softmax(output,dim=-1)
        loss=(-log_preds.sum(dim=-1)).mean()
        nll=F.nll_loss(log_preds,target)
        final_loss=self.e*loss/num_classes+(1-self.e)*nll
        return  final_loss

#FocalLoss
def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

# ArcLoss,暂时不用
class ArcLoss(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)

    def forward(self, feature, m=0.5, s=30):
        x = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.W, dim=0)
        cos = torch.matmul(x, w)/10             # 求两个向量夹角的余弦值
        a = torch.acos(cos)                     # 反三角函数求得 α
        top = torch.exp(s*torch.cos(a+m))       # e^(s * cos(a + m))
        down2 = torch.sum(torch.exp(s*torch.cos(a)), dim=1, keepdim=True)-torch.exp(s*torch.cos(a))
        out = torch.log(top/(top+down2))
        return out


#PolyCELoss
class PolyCE(nn.Module):
    def __init__(self,class_number=2,weight=None):
        super(PolyCE, self,).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.epsilon=1.0
        self.classnum=class_number

    def forward(self, input, target):

        poly = torch.sum(F.one_hot(target, self.classnum).float() * F.softmax(input,dim=1), dim=-1)
        ce_loss=self.ce(input, target)
        poly_ce_loss = ce_loss + self.epsilon * (1 - poly)
        return poly_ce_loss.mean()

#PolyFocalLoss
class PolyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_number=2,weight=None):
        super( PolyFocalLoss, self, ).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.classnum = class_number
        self.weight=weight

    def forward(self, input, target):
        focal_loss_func = FocalLoss(gamma=self.gamma,weight=self.weight)
        focal_loss = focal_loss_func(input, target)

        p = torch.sigmoid(input)
        labels = torch.nn.functional.one_hot(target, self.classnum)
        # labels = torch.tensor(labels, dtype=torch.float32)
        labels=labels.clone()
        poly = labels * p + (1 - labels) * (1 - p)
        poly_focal_loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly, 2 + 1), dim=-1)
        return poly_focal_loss.mean()

#LDAMLoss
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_s = batch_m.view((-1, 1))
        x_m = x - batch_s

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
