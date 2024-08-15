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
    parser.add_argument('--model_path',default=r'./pt/T15/best.pt')
    parser.add_argument('--imagefile', default=r'./5_LMPS')
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


###{'cat': 0, 'dog': 1}
#载入标签
with open('./pt/lable.plk','rb') as f:
    label=pickle.load(f)
    label=dict((v,k) for k,v in label.items())

imagepaths=list(paths.list_images(args.imagefile))
pic_nums=len(imagepaths)
output=r'./batch_size2'
batch_size=2

epochs=pic_nums//batch_size

if not os.path.exists(output):
    os.makedirs(output)



# 载入模型训练权重
net=Net('convnext_pico.d1_in1k',num_class=9,pretrained=False,embeddingdim=256,mode='pred').to(device)
net.load_state_dict(torch.load(args.model_path,map_location=device))

mean=(127.5,127.5,127.5)
std=(127.5,127.5,127.5)

n=0
for epoch in tqdm(range(epochs)):

    names=[]
    img_bs = []
    pics=[]
    for i in range(batch_size):
        imagepath=imagepaths[n]
        imagebak = cv2.imread(imagepath)
        pics.append(imagebak)
        image=imagebak[:,:,::-1]
        basename = os.path.basename(imagepath)
        name = basename.split('.')[0]
        names.append(name)

        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32)  # 注意输入type一定要np.float32
        image -= mean
        image /= std
        image = np.transpose(image, (2, 0, 1))
        img_bs.append(image)
        n+=1
    img_bs=np.array(img_bs)
    img_bs = torch.from_numpy(img_bs)

    with torch.no_grad():
        net.eval()
        index, score=net(img_bs)
        indexs = index.data.cpu().numpy()
        scores = score.data.cpu().numpy()

        for j in range(batch_size):
            index=indexs[j]
            score=scores[j]
            outputout=os.path.join(output,'cv2_{}_{}_{:.2f}%.png'.format(names[j],label[int(index)],score * 100))
            cv2.imwrite(outputout,pics[j])















