#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from scipy.special import softmax
import pickle
import cv2
onnx_path=r'./onnx/best-smi.onnx'
pic_path=r'./onnx/22.bmp'
w,h=112,112

mean=(127.5,127.5,127.5)
std=(127.5,127.5,127.5)

model = onnx.load(onnx_path)
onnx.checker.check_model(model)

session = ort.InferenceSession(onnx_path)

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


outputs = session.run(None,input_feed = { 'input' : img })
index, score=outputs[0],outputs[1]
print('{}:{:.4f}'.format(label[int(index)],float(score)))



