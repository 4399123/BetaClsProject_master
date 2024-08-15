#encoding=gbk
import numpy as np
import pickle
import cv2
import torch

pt_path=r'./torchscript/clsslib.pt'
pic_path=r'./torchscript/11.bmp'
w,h=112,112

mean=(127.5,127.5,127.5)
std=(127.5,127.5,127.5)

net=torch.jit.load(pt_path)
net.eval()
net.cuda()

#载入标签
with open('./pt/lable.plk','rb') as f:
    label=pickle.load(f)
    label=dict((v,k) for k,v in label.items())

#输入图像预处理
img=cv2.imread(pic_path)
img=cv2.resize(img,(w,h))
img=img[:,:,::-1]
img = np.array(img).astype(np.float32)  # 注意输入type一定要np.float32
img -= mean
img /= std
img = np.array([np.transpose(img, (2, 0, 1))])
img=torch.from_numpy(img).cuda()


index, score = net(img)
index = index.data.cpu().numpy()[0]
score = score.data.cpu().numpy()[0]

print('{}:{:.4f}'.format(label[int(index)],float(score)))



