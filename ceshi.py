import cv2
from PIL import Image

path=r'D:\blueface\Cls_BlueFce\CTimmProject\image\dog.jpg'

img=cv2.imread(path)
cv2.imshow('111',img)


img2=Image.open(path)
img2.show()

cv2.waitKey(0)