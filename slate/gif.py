import numpy as np
from PIL import Image
import sys
import cv2
from natsort import natsorted
from tqdm import tqdm
import os

def load_data(path_num):
    if (path_num==0) or (path_num=='before'):
        path = 'rgb/before/'
    elif (path_num==1) or (path_num=='after'):
        path = 'rgb/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = 'rgb/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)

    temp_img = cv2.imread(path+pic_paths[0],cv2.COLOR_BGR2RGB) 
    pics_without_line = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1],temp_img.shape[2]))
    # pics_without_line = []
    for i,j in enumerate(pic_paths):
        aa = cv2.imread(path+j,cv2.COLOR_BGR2RGB)
        pics_without_line[i] = aa.copy()
    pics_without_line = pics_without_line.astype(np.float32)
    return pics_without_line

path_num = sys.argv[1]

imgs = (load_data(path_num)*255).astype(np.uint8)
imgs = [Image.fromarray(img) for img in imgs]
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save(f"{path_num}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)