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
from utils.net import Net

# 用于命令项选项与参数解析
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',default=r'./pt/22ppm_x5/efficientnet_b5.pt')
    parser.add_argument('--imagefile', default=r'./image')
    # parser.add_argument('--n', default=5)
    args=parser.parse_args()
    return args

args=get_args()

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

if not os.path.exists('./out'):
    os.makedirs('./out')


###{'cat': 0, 'dog': 1}
#载入标签
with open('./pt/lable.plk','rb') as f:
    label=pickle.load(f)
    label=dict((v,k) for k,v in label.items())

imagepaths=list(paths.list_images(args.imagefile))

# 载入模型训练权重
net=Net('tf_efficientnet_b5.ns_jft_in1k',num_class=15,pretrained=False,embeddingdim=1024,mode='pred').to(device)
net.load_state_dict(torch.load(args.model_path,map_location=device))

mean=(127.5,127.5,127.5)
std=(127.5,127.5,127.5)
nn=0
for imgpath in tqdm(imagepaths):
    basename=os.path.basename(imgpath)
    name=basename.split('.')[0]
    image = cv2.imread(imgpath)
    try:
        imgbak = image.copy()
    except:
        continue
    image=image[:,:,::-1]
    image=cv2.resize(image,(112,112))
    image = np.array(image).astype(np.float32)  # 注意输入type一定要np.float32
    image-=mean
    image/=std
    image= np.array([np.transpose(image, (2, 0, 1))])
    image=torch.from_numpy(image)
    with torch.no_grad():
        net.eval()
        index, score=net(image)
        index = index.data.cpu().numpy()[0]
        score =  score.data.cpu().numpy()[0]
        # cv2.imwrite('./out/cv2_{}_{}_{:.2f}%.png'.format(name,label[int(index)],score * 100),imgbak)
        cv2.imwrite('./out/{}_{}_{:.2f}%.png'.format(nn, label[int(index)], score * 100), imgbak)
        nn+=1















