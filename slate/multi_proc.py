import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as scp
from natsort import natsorted
import cv2
import multiprocessing
from numpy.fft import fft2,fft,ifft


# from itertools import repeat


# Run the script using python testing.py 0/1/2
# 0 means before, 1 means after, 2 means after2min data

def num_frames(path_num):
    if (path_num==0) or (path_num=='before'):
        path = '../pig_data/before/'
    elif (path_num==1) or (path_num=='after'):
        path = '../pig_data/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = '../pig_data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    return len(range(0,len(pic_paths)-60,2))

def load_data(path_num,range_frames):
    if (path_num==0) or (path_num=='before'):
        path = '../pig_data/before/'
    elif (path_num==1) or (path_num=='after'):
        path = '../pig_data/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = '../pig_data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    pic_paths = pic_paths[range_frames:range_frames+60]

    temp_img = cv2.imread(path+pic_paths[0],cv2.IMREAD_UNCHANGED) 
    pics_without_line = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1]))
    # pics_without_line = []
    for i,j in enumerate(pic_paths):
        aa = cv2.imread(path+j,cv2.IMREAD_UNCHANGED)
        pics_without_line[i]=(aa.copy())
    pics_without_line = pics_without_line.astype(np.float32)
    return pics_without_line


def min_max(data1):
    if np.max(data1)==0:
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1


# def autocorr(slope_arr):
#     mask1 = np.zeros_like(slope_arr[0],dtype=np.float32)
#     std_mask = np.apply_along_axis(func1d=np.std,arr=slope_arr,axis=0)
#     for x in range(slope_arr.shape[1]):
#         for y in range(slope_arr.shape[2]):
#             data1 = slope_arr[:,x,y]
#             corr = acf(data1)
#             slope1 = np.polyfit(range(len(corr)), corr, 1)[0]
#             mask1[x, y] = -np.abs(slope1)
#     return mask1*(2*std_mask)

# def fft_slope(slope_arr):
#     mask1 = np.zeros_like(slope_arr[0],dtype=np.float32)
#     std_mask = np.apply_along_axis(func1d=np.std,arr=slope_arr,axis=0)
#     slope_arr = np.apply_along_axis(func1d=min_max,arr=slope_arr,axis=0)
#     for x in range(slope_arr.shape[1]):
#         for y in range(slope_arr.shape[2]):
#             data1 = slope_arr[:,x,y]
#             fft_data = np.abs(fft(data1))
#             slope1 = np.polyfit(range(len(fft_data)), fft_data, 1)[0]
#             mask1[x, y] = -np.abs(slope1)
#     return mask1*(2*std_mask)

# def corr_fft(img):
#     height, width = img.shape[1], img.shape[2]
#     padded_img = np.pad(img, ((0, 0), (5, 5), (5, 5)), mode='constant', constant_values=0)
#     mask_image2 = np.zeros((height, width))
#     for y in range(height):
#         for x in range(width):
#             correlations = []
#             abs_fft_patch_first = np.abs(fft2(min_max(padded_img[0, y:y+10, x:x+10]))).flatten()
#             for i in range(40):
#                 for kk in range(padded_img.shape[0]):
#                     padded_img[kk] = min_max(padded_img[kk])
#                 patch = padded_img[i, y:y+10, x:x+10]
#                 fft_patch = fft2(patch)
#                 abs_fft_patch = np.abs(fft_patch).flatten()
#                 corr = np.correlate(abs_fft_patch, abs_fft_patch_first)
#                 correlations.append(corr[0])
#             mask_image2[y, x] = np.mean(correlations)
#     mask_image2 = -mask_image2
#     mask_image2 = mask_image2-np.min(mask_image2)
#     return (mask_image2)

def slope_mask_10batch(slope_arr):
    mask1 = np.zeros_like(slope_arr[0],dtype=np.float32)
    # slope_arr = slope_arr.astype(np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=slope_arr,axis=0)
    slope_arr = np.apply_along_axis(func1d=min_max,arr=slope_arr,axis=0)
    for x in range(slope_arr.shape[1]):
        for y in range(slope_arr.shape[2]):
            data1 = slope_arr[:,x,y]
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*(5*std_mask)

def image(x_idx_batch, X_shape, shared_array):
    data1 = shared_array.astype(np.float32)
    # mask_temp = autocorr(data1)
    # mask_temp = corr_fft(data1)
    # mask_temp = fft_slope(data1)
    # mask_temp = opt_flow(data1)
    mask_temp = slope_mask_10batch(data1)
    return mask_temp , x_idx_batch


if __name__ == '__main__':
    # num = int(sys.argv[1])
    data_name = str(sys.argv[2])
    batch = int(sys.argv[1])

    tot_num_frames = num_frames(data_name)

    if tot_num_frames-batch<120:
        arr = np.zeros((tot_num_frames-batch,40,1000,954),dtype=np.float32)
    else:
        arr = np.zeros((120,60,1000,954),dtype=np.float32)
    
    j=0
    for num in range(batch,batch+120):
        arr[j] = load_data(data_name,num*2)
        if j==arr.shape[0]-1:
            break
        else:
            j+=1
        
    arr = arr.astype(np.float32)
    for i in range(arr.shape[0]):
        arr[i] = arr[i]/np.max(arr[i])
    # arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)

    mask_each_batch = np.zeros((arr.shape[0],arr.shape[2],arr.shape[3]), dtype=np.float32)
    X_shape = (arr.shape[0], arr.shape[1], arr.shape[2],arr.shape[3]) # 120x60x1000x954
    X = multiprocessing.Array('f', X_shape[0] * X_shape[1] * X_shape[2] * X_shape[3], lock=False)

    X_np = np.frombuffer(X, dtype=np.float32).reshape(X_shape)
    np.copyto(X_np, arr)
    del arr

    with multiprocessing.Pool(processes=120) as pool:
        results = [pool.apply_async(image, args=(x_num, X_shape, X_np[x_num])) for x_num in range(X_shape[0])]
        pool.close()
        pool.join()

    for r in results:
        m,x_idx = r.get()
        mask_each_batch[x_idx] = m


    with open(f'tmp/{data_name}_mask_{batch}.pickle', 'wb') as handle:
        pickle.dump(mask_each_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

