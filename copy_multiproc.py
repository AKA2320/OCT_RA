import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as scp
from natsort import natsorted
import cv2
import multiprocessing
# from itertools import repeat


# Run the script using python testing.py 0/1/2
# 0 means before, 1 means after, 2 means after2min data

def load_data(path_num,range_frames=None):
    if (path_num==0) or (path_num=='sc1_part1'):
        path = '../data/cow_sc1_part1/'
    elif (path_num==1) or (path_num=='sc1_part2'):
        path = '../data/cow_sc1_part2/'
    elif (path_num==2) or (path_num=='sc2'):
        path = '../data/cow_sc2/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    if range_frames:
        pic_paths = pic_paths[range_frames-20:range_frames+20]
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

def initpool(mask_arr):
    global X_mask
    X_mask = mask_arr

def image(x, y, X_shape):
    X_np = np.frombuffer(X_mask.get_obj(), dtype=np.float32).reshape(X_shape)
    data1 = X_np[:,x,y].astype(np.float32)
    std_mask = np.std(data1)
    data1 = min_max(data1)
    slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
    return x, y,((-np.abs(slope1))*(4*std_mask))


if __name__ == '__main__':
    num = int(sys.argv[1])
    data_name = str(sys.argv[2])

    arr = load_data(data_name,num)
    arr = arr.astype(np.float32)
    mask1 = np.zeros_like(arr[0], dtype=np.float32)
    # std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)   
    # arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)

    X_shape = (arr.shape[0], arr.shape[1], arr.shape[2],arr.shape[3]) # 120x40x1500x454
    X = multiprocessing.Array('f', X_shape[0] * X_shape[1] * X_shape[2] * X_shape[3], lock=False)
    # Pa_shape = multiprocessing.Array('i', X_shape)#to share the shape data
    X_np = np.frombuffer(X.get_obj(), dtype=np.float32).reshape(X_shape)
    np.copyto(X_np, arr)
    pool = multiprocessing.Pool(processes=os.cpu_count(), initializer=initpool, initargs=(X,))
    res = [pool.apply_async(image, args=(x, y, X_shape)) for x in range(arr.shape[0]) for y in range(arr.shape[2])]
    pool.close()
    pool.join()
    for r in res:
        x, y, value = r.get()
        mask1[x, y] = value
    
    # mask = mask1

    with open(f'tmp/{data_name}_mask_{num}.pickle', 'wb') as handle:
        pickle.dump(mask1, handle, protocol=pickle.HIGHEST_PROTOCOL)

