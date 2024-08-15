from imutils import  paths
import os
from tqdm import tqdm

path=r'./select_imgs'
n=0

imgpaths=list(paths.list_images(path))

for imgpath in tqdm(imgpaths):
    name = '1023BMQQ.bmp'
    name='{}_{}'.format(n,name)
    dirname=os.path.dirname(imgpath)
    os.rename(imgpath,os.path.join(dirname,name))
    n+=1

