# -*- coding: UTF-8 -*-
import time
import argparse
from timm.utils import model_ema
import torch.nn as nn
from torchsummary import summary
import torch
from timm.optim import AdamW,AdamP,Lamb
from torch.optim import SGD
from timm.scheduler import CosineLRScheduler
import timm
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
from utils.zutil import get_log,get_device_info,data_preprocessing,ModelEmaV2,TimeMeter
from utils.loss import get_lossfunc
from utils.net import Net
from pytorch_metric_learning import losses,regularizers,miners
# from timm.models import convnext
from tqdm import tqdm
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser=argparse.ArgumentParser("classification")
parser.add_argument('--pathtrain',default=r'../BlueFaceCls22PPM_V8/train',help='train data')
parser.add_argument('--pathval',default=r'../BlueFaceCls22PPM_V8/val',help='val data')
parser.add_argument('--bs',default=2508,type=int,help='batchsize')
parser.add_argument('--lr',default=0.0003)
parser.add_argument('--epochs',default=800)
parser.add_argument('--loss',default='1',type=int,help='损失函数种类:0-4')
parser.add_argument('--embeddingdim',default=1024,type=int,help='输出特征维度')
parser.add_argument('--finetune-from', type=str, default=None,)
parser.add_argument('--arcloss', default=True, type=bool, help='使用arcfaceloss辅助训练')
parser.add_argument('--model_ema',default=True,type=bool,help='是否使用权重指数平均移动')
parser.add_argument('--checkpoint',default=False,type=bool,help='是否使用断点训练')
parser.add_argument('--checkpoint_path',default='./pt/1.pt',help='需要继续训练的模型路径')
args=parser.parse_args()

w,h=112,112
#构造tensorboard 写入容器
writer=SummaryWriter()

# #训练日志保存
logger=get_log(args)

#启用CPU或GPU,获取设备信息
device,accelerator=get_device_info()

#数据预处理
train_loader,val_loader,train_dataset,val_dataset=data_preprocessing(args,h,w,accelerator)
num_classes=len(train_dataset.classes)

#保存标签到pickle
with open('./pt/lable.plk','wb') as f:
    pickle.dump(train_dataset.class_to_idx,f)

## meters
time_meter= TimeMeter(args.epochs)

net=Net('convnext_pico.d1_in1k',num_class=num_classes,pretrained=False,embeddingdim=args.embeddingdim,mode='train').to(device)
# net=Net('convnext_nano.d1h_in1k',num_class=num_classes,pretrained=False,embeddingdim=args.embeddingdim,mode='train').to(device)
# net=Net('convnextv2_pico.fcmae_ft_in1k',num_class=num_classes,pretrained=False,embeddingdim=args.embeddingdim,mode='train').to(device)
# net=Net('tf_efficientnet_b2.ns_jft_in1k',num_class=num_classes,pretrained=False,embeddingdim=args.embeddingdim,mode='train').to(device)

model_ema=None
if args.model_ema:
    model_ema=ModelEmaV2(net)
# if args.model_ema:
#     ModelEmaV2(net)

if not args.finetune_from is None:
    logger.info(f'load pretrained weights from {args.finetune_from}')
    msg = net.load_state_dict(torch.load(args.finetune_from,map_location='cpu'), strict=False)
    logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
    logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
net.to(device)

# loss选择
criterion = get_lossfunc(args,train_dataset,accelerator)

if args.arcloss:
    miner_func = miners.MultiSimilarityMiner()
    arcloss=losses.SubCenterArcFaceLoss(num_classes=num_classes,embedding_size=args.embeddingdim,weight_regularizer=regularizers.RegularFaceRegularizer()).to(device)
    optimizer = AdamP([{'params':net.parameters()},{'params':arcloss.parameters()}], lr=args.lr, weight_decay=5e-4, nesterov=True)
else:
    optimizer = AdamP(net.parameters(),lr=args.lr,weight_decay=5e-4,nesterov=True)

# 训练加速器加载
net, optimizer, train_loader = accelerator.prepare(net, optimizer, train_loader)
schedule=CosineLRScheduler(optimizer=optimizer,
                           t_initial=args.epochs,
                           lr_min=9.5e-5,
                           warmup_t=5,
                           warmup_lr_init=1e-4)
time.sleep(2)
# summary(net,(3,h,w))

def train(epoch):
    net.train()
    batch_imgs=0
    batch_correct_imgs=0
    batch_train_loss=0
    for batch_idx,data in enumerate(train_loader):
        inputs,targets=data
        inputs=inputs.to(device)
        targets=targets.to(device)

        optimizer.zero_grad()
        feature,output=net(inputs)
        loss=criterion(output,targets).to(device)
        if args.arcloss:
            miner_output = miner_func(feature,targets)
            arcfaceloss=arcloss(feature,targets,miner_output)
            loss+=0.1*arcfaceloss
        accelerator.backward(loss)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(net)

        #将lr写入tensorbaord容器
        lr=optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', lr, epoch)

        batch_train_loss=loss.item()
        batch_idx+=1
        batch_imgs=targets.size(0)

        #根据数据集大小训练过程展示的间隔
        gap=int(len(train_dataset.imgs)/(10*args.bs))
        if(gap==0):gap=2
        if(batch_idx%gap==0):
            _,predicted=torch.max(output.data,dim=1)
            batch_correct_imgs=predicted.eq(targets).sum().item()
            logger.info('epoch:{}/{},batch:{}/{},lr:{:.6f},train loss:{:.5f},train acc:{:.2f}%'.format(epoch+1,args.epochs,
                                                                                     batch_idx,
                                                                                     len(train_loader),
                                                                                     lr,
                                                                                     batch_train_loss,
                                                                                     batch_correct_imgs/batch_imgs*100))

    #将训练损失和精度写入tensorboard容器
    writer.add_scalar('loss/train_loss', batch_train_loss, epoch)
    writer.add_scalar('acc/train_acc', batch_correct_imgs/batch_imgs, epoch)

    interv, ets = time_meter.get()
    logger.info('ets:{},interv:{:.2f}s'.format(ets, interv))
    time_meter.update()

def val(epoch):

    net.eval()
    correct = 0
    sumimg = 0
    with torch.no_grad():
        for batch_idx,data in tqdm(enumerate(val_loader)):
            inputs,targets=data
            inputs=inputs.to(device)
            targets=targets.to(device)

            _,output=net(inputs)
            _,predicted=torch.max(output.data,dim=1)
            correct+=predicted.eq(targets).sum().item()
            sumimg+=targets.size(0)
        val_acc=correct/sumimg*100

        #将验证精度写入tensorboard容器
        writer.add_scalar('acc/val_acc',val_acc,epoch)

        logger.info('【验证精度val_acc】:{:.3f}%,【correct/total】:{}/{}'.format(val_acc,correct,sumimg))
        return val_acc

acc=0
# 断点训练
start_epoch = 0
if args.checkpoint:
    print('启动断点训练!')
    path_checkpoint = args.checkpoint_path
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

for epoch in range(start_epoch,args.epochs):

    schedule.step(epoch)
    train(epoch)
    val_acc=val(epoch)
    if(val_acc>=acc):
        acc=val_acc

        #用于断点保存
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, './pt/{:.2f}.pt'.format(acc))
        torch.save(net.state_dict(),'./pt/best.pt')
        logger.info("save model!!!")
logger.info("ending!!!")





