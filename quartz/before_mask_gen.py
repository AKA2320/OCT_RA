import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from scipy import ndimage as scp
from natsort import natsorted
import cv2

def load_data(path_num,range_frames=None):
    if (path_num==0) or (path_num=='before'):
        path = '../../data/before/'
    elif (path_num==1) or (path_num=='after'):
        path = '../../data/after/'
    elif (path_num==2) or (path_num=='after2min'):
        path = '../../data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    if range_frames:
        pic_paths = pic_paths[range_frames-20:range_frames+20]
    pics_without_line = []

    for i in pic_paths:
        aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
        pics_without_line.append(aa.copy())

    pics_without_line = np.array(pics_without_line).astype(np.float32)
    zero_line_down= []
    zero_line_up = []
    zero_line_left = []
    zero_line_right = []
    for i in range(pics_without_line.shape[0]):
        for down in range(pics_without_line[i].shape[0]-1,-1,-1):
            if np.any(pics_without_line[i][down,:]!=0):
                zero_line_down.append(down)
                break
        for up in range(0,pics_without_line[i].shape[0]):
            if np.any(pics_without_line[i][up,:]!=0):
                zero_line_up.append(up)
                break
        for left in range(0,pics_without_line[i].shape[1]):
            if np.any(pics_without_line[i][:,left]!=0):
                zero_line_left.append(left)
                break
        for right in range(pics_without_line[i].shape[1]-1,-1,-1):
            if np.any(pics_without_line[i][:,right]!=0):
                zero_line_right.append(right)
                break
    zero_line_down = np.min(zero_line_down)
    zero_line_up = np.min(zero_line_up)
    zero_line_left = np.min(zero_line_left)
    zero_line_right = np.min(zero_line_right)
    pics_without_line[:,zero_line_down:,:] =  0
    pics_without_line[:,:zero_line_up,:] =  0
    pics_without_line[:,:,:zero_line_left] =  0
    pics_without_line[:,:,zero_line_right:] =  0
    return pics_without_line

def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1

def slope_mask_10batch(arr,p1):
    mask1 = np.zeros_like(arr[0],dtype=np.float32)
    arr = arr.astype(np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)
    arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
    for x in range(arr.shape[1]):
        for y in range(arr.shape[2]):
            data1 = arr[:,x,y].astype(np.float32).copy()
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*std_mask

num = int(sys.argv[1])
data_name = str(sys.argv[2])
data = load_data(data_name,num)

data = data.astype(np.float32)
mask = np.zeros((data.shape[1],data.shape[2]),dtype=np.float32)
mask = slope_mask_10batch(data,num)


with open(f'tmp/{data_name}_mask_{num}.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

