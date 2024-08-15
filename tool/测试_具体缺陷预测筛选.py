import os
from imutils import paths
from tqdm import tqdm
import shutil


path=r'./out'
output=r'./output'

if not os.path.exists(output):
    os.makedirs(output)

imgpaths=list(paths.list_images(path))

for imgpath in tqdm(imgpaths):
    basename=os.path.basename(imgpath)
    lable=basename.split('_')[2]
    if('MNYW' not in lable):
        shutil.copy(imgpath,os.path.join(output,basename))

