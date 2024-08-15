import logging
import torch
from torchvision import datasets,transforms
from torchvision.transforms import autoaugment
from torch.utils.data  import DataLoader
from accelerate import Accelerator
import torch.nn as nn
import warnings
import numpy as np
from copy import deepcopy
from pytorch_metric_learning import samplers
import time
import datetime
from torchtoolbox.transform import Cutout
# #训练日志保存
def get_log(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handerler = logging.StreamHandler()
    logger.addHandler(stream_handerler)
    file_hander = logging.FileHandler('train.log')
    logger.addHandler(file_hander)
    logger.info(args)
    return logger

#启用CPU或GPU,获取设备信息,构造加速器,开启混合精度训练
def get_device_info():

    if torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision='fp16')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # accelerator = Accelerator(mixed_precision='bf16')
        torch.backends.cudnn.benchmark = True
        accelerator.print('启用GPU加速训练！')
        p = torch.cuda.get_device_properties(0)
        accelerator.print('{},GPU:{},total_memory:{:.2f}G'.format(torch.__version__, p.name, p.total_memory / 1024 ** 3))
        device = accelerator.device
    else:
        accelerator = Accelerator()
        torch.set_default_tensor_type('torch.FloatTensor')
        accelerator.print('启用CPU训练！')
        accelerator.print('{}'.format(torch.__version__))
        device = accelerator.device
    return device,accelerator


def data_preprocessing(args,h,w,accelerator):
    data_transform_train = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        # Cutout(),
        # autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy("imagenet")),
        autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.448, 0.455, 0.417], std=[0.226, 0.221, 0.221])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_transform_val = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    labels=list()
    # 输入数据预处理
    train_dataset = datasets.ImageFolder(root=args.pathtrain, transform=data_transform_train)

    # 采样器设置
    for label in train_dataset.imgs:
        labels.append(int(label[1]))
    sampler = samplers.MPerClassSampler(labels,
                                        m=int(args.bs//len(train_dataset.classes)),
                                        batch_size=args.bs,
                                        length_before_new_iter=len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs,sampler=sampler)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs)
    val_dataset = datasets.ImageFolder(root=args.pathval, transform=data_transform_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs)
    accelerator.print(train_dataset.class_to_idx)
    accelerator.print('image numbers:{}'.format(len(train_dataset.imgs)))
    return train_loader,val_loader,train_dataset,val_dataset


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)



class TimeMeter(object):

    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.st = time.time()
        self.global_st = self.st
        self.curr = self.st

    def update(self):
        self.iter += 1

    def get(self):
        self.curr = time.time()
        interv = self.curr - self.st
        global_interv = self.curr - self.global_st
        eta1 = int((self.max_iter-self.iter) * (global_interv / (self.iter+1)))
        eta = str(datetime.timedelta(seconds=eta1))
        self.st = self.curr
        return interv, eta



