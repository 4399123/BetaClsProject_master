import os.path

import cv2
import torch
import torch.nn.functional as F
# from model.TFNet import Model
import numpy as np
from imutils import paths
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
import timm
import pickle
from utils.netbak1 import Net
import shutil

# 用于命令项选项与参数解析
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',default=r'./pt/T8/best.pt')
    parser.add_argument('--imagefile', default=r'C:\ZL\Cls7\cls_imgs')
    args=parser.parse_args()
    return args

args=get_args()

outpath=r'C:\ZL\Cls7\img_select'
if not os.path.exists(outpath):
    os.makedirs(outpath)

background_path=r'C:\ZL\Cls7\img_select\0_background'
if not os.path.exists(background_path):
    os.makedirs(background_path)

BMQQ_path=r'C:\ZL\Cls7\img_select\6_BMQQ'
if not os.path.exists(BMQQ_path):
    os.makedirs(BMQQ_path)

KTAK_path=r'C:\ZL\Cls7\img_select\8_KTAK'
if not os.path.exists(KTAK_path):
    os.makedirs(KTAK_path)

LMHH_path=r'C:\ZL\Cls7\img_select\7_LMHH'
if not os.path.exists(LMHH_path):
    os.makedirs(LMHH_path)

LMPS_path=r'C:\ZL\Cls7\img_select\5_LMPS'
if not os.path.exists(LMPS_path):
    os.makedirs(LMPS_path)

MDBD_path=r'C:\ZL\Cls7\img_select\2_MDBD'
if not os.path.exists(MDBD_path):
    os.makedirs(MDBD_path)

MNYW_path=r'C:\ZL\Cls7\img_select\3_MNYW'
if not os.path.exists(MNYW_path):
    os.makedirs(MNYW_path)

QPZZ_path=r'C:\ZL\Cls7\img_select\1_QPZZ'
if not os.path.exists(QPZZ_path):
    os.makedirs(QPZZ_path)

WW_path=r'C:\ZL\Cls7\img_select\4_WW'
if not os.path.exists(WW_path):
    os.makedirs(WW_path)


#构造加速器
accelerator = Accelerator()

#启用CPU或GPU,获取设备信息
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    print('启用GPU加速推理！')
    p = torch.cuda.get_device_properties(0)
    pp = '{},GPU:{},total_memory:{:.2f}G'.format(torch.__version__, p.name, p.total_memory / 1024 ** 3)
    print(pp)
    device = accelerator.device
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print('启用CPU推理！')
    pp = '{}'.format(torch.__version__)
    print(pp)
    device = accelerator.device



with open('./pt/lable.plk','rb') as f:
    label=pickle.load(f)
    label=dict((v,k) for k,v in label.items())

imagepaths=list(paths.list_images(args.imagefile))

# 载入模型训练权重
net=Net('convnext_pico.d1_in1k',num_class=9,pretrained=False,embeddingdim=512).to(device)
net.load_state_dict(torch.load(args.model_path,map_location=device))

mean=(127.5,127.5,127.5)
std=(127.5,127.5,127.5)
for imgpath in tqdm(imagepaths):
    basename=os.path.basename(imgpath)
    name=basename.split('.')[0]
    image = cv2.imread(imgpath)
    imgbak = image.copy()
    image=cv2.resize(image,(112,112))
    image = np.array(image).astype(np.float32)  # 注意输入type一定要np.float32
    image-=mean
    image/=std
    image= np.array([np.transpose(image, (2, 0, 1))])
    image=torch.from_numpy(image)
    with torch.no_grad():
        net.eval()
        feature,output= net(image)
        score, prediect = torch.max(F.softmax(output), dim=1)
        score = score.data.cpu().numpy()[0]
        prediect = prediect.data.cpu().numpy()[0]
        # cv2.imwrite('./out/2{}_{}_{:.2f}%.png'.format(name,label[prediect],score * 100),imgbak)
        pre=label[prediect]
        if(pre=='6_BMQQ'):
            shutil.copy(imgpath,os.path.join(BMQQ_path,basename))
        elif(pre=='8_KTAK'):
            shutil.copy(imgpath, os.path.join(KTAK_path, basename))
        elif(pre=='7_LMHH'):
            shutil.copy(imgpath, os.path.join(LMHH_path, basename))
        elif(pre=='5_LMPS'):
            shutil.copy(imgpath, os.path.join(LMPS_path, basename))
        elif(pre=='2_MDBD'):
            shutil.copy(imgpath, os.path.join(MDBD_path, basename))
        elif(pre=='3_MNYW'):
            shutil.copy(imgpath, os.path.join(MNYW_path, basename))
        elif(pre=='1_QPZZ'):
            shutil.copy(imgpath, os.path.join(QPZZ_path, basename))
        elif(pre=='4_WW'):
            shutil.copy(imgpath, os.path.join(WW_path, basename))
        elif(pre=='0_background'):
            shutil.copy(imgpath, os.path.join(background_path, basename))
        else:
            print('error_{}'.format(pre))















