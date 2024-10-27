
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os

import cv2
from natsort import natsorted
from scipy import ndimage as scp
from tqdm import tqdm
import pickle
import time
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist
import sys

num = int(sys.argv[1])
name = str(sys.argv[2])


def load_data(path_num,range_frames,dis=False):
    if (path_num==0) or (path_num=='before'):
        path = 'pig_eyeball/registered/before/'
    elif (path_num==1) or (path_num=='after'):
        path = 'pig_eyeball/registered/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = 'pig_eyeball/registered/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG') or i.endswith('.tiff'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    pic_paths = pic_paths[range_frames:range_frames+60]
    pics_without_line = []

    for i in tqdm(pic_paths,desc='Loading data',disable=dis):
        aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
        # aa = dicom.dcmread(path+i).pixel_array
        pics_without_line.append(aa.copy())
    pics_without_line = np.array(pics_without_line).astype(np.float32)
    return pics_without_line


def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1

def gen_mean_img(path_num,range_frames,dis):
    mean_img = load_data(path_num,range_frames,dis)
    mean_img = (mean_img-np.min(mean_img))/(np.max(mean_img)-np.min(mean_img))
    mean_img = equalize_adapthist(np.mean(mean_img,axis=0),nbins=4000)
    return mean_img

all_after_mean = np.zeros((len(range(0,2500-60,2)),1000,954))
j=0
for i in tqdm(range(0,2500-60,2)):
    all_after_mean[j] = gen_mean_img(num,i,dis=False)
    j+=1
    
# all_after_mean = all_after_mean[:,70:-70,70:-70]
with open(f'pig_60_mean_{name}.pickle', 'wb') as handle:
    pickle.dump(all_after_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

