

import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from scipy import ndimage as scp
from natsort import natsorted
import cv2

# Run the script using python testing.py 0/1/2
# 0 means before, 1 means after, 2 means after2min data

def load_data(path_num,range_frames=None):
    if path_num==0:
        path = '../../data/before/'
    elif path_num==1:
        path = '../../data/after/'
    elif path_num==2:
        path = '../../data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    if range_frames:
        pic_paths = pic_paths[range_frames-50:range_frames+50]
    pics_without_line = []

    for i in pic_paths:
        aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
        pics_without_line.append(aa.copy())

    pics_without_line = np.array(pics_without_line).astype(np.float32)
    return pics_without_line


def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1

def slope_mask_10batch(arr,p1):
    mask1 = np.zeros_like(arr[0],dtype=np.float32)
    arr = arr[p1-50:p1+50,:,:].astype(np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)
    arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
    for x in range(arr.shape[1]):
        for y in range(arr.shape[2]):
            data1 = arr[:,x,y].astype(np.float32).copy()
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*std_mask

num = int(sys.argv[1])
data = load_data(num)

data = data.astype(np.float32)
mask = np.zeros((len(range(50,2500-50,2)),data.shape[1],data.shape[2]),dtype=np.float32)
j=0
for i in range(50,2500-50,2):
    mask[j] = slope_mask_10batch(data,i)
    j+=1


with open(f'tmp/mask_{num}.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)