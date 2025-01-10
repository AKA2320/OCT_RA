import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as scp
from statsmodels.tsa.stattools import acf
from natsort import natsorted
import cv2
import multiprocessing
from numpy.fft import fft2,fft,ifft
# from itertools import repeat
from utils.py import *

os.chdir('../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/')

def load_data(path):
    pic_paths = []
    for scan_num in os.listdir(path):
        pic_paths.append(os.path.join(path,scan_num,f'{scan_num}.pickle'))
    pic_paths = natsorted(pic_paths)

    with open(f'{pic_paths[0]}', 'rb') as handle:
        b = pickle.load(handle)
    data = np.zeros((len(pic_paths),b.shape[0],b.shape[1],b.shape[2]))

    for idx,img_path in enumerate(pic_paths):
        temp = pickle.load(img_path)
        data[i]=(temp.copy())
    data = data.astype(np.float32)
    return data


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

def ymotion(data):
    n = data.shape[0]
    data[0,0,250:500,:]
    nn = [np.argmax(np.sum(data[i][0,250:500,:],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j]  = warp(data[i][j],AffineTransform(matrix=tf_all_nn[i]),order=3)
    return data

# def image(x_idx_batch, X_shape, shared_array):
#     data1 = shared_array.astype(np.float32)
#     # mask_temp = autocorr(data1)
#     # mask_temp = corr_fft(data1)
#     mask_temp = fft_slope(data1)
#     # mask_temp = slope_mask_10batch(data1)
#     return mask_temp , x_idx_batch


if __name__ == '__main__':
    # num = int(sys.argv[1])
    path = 'timelapse_withH2O2_registered/registered_pickles/'
    data = load_data(path)
    data = ymotion(data)

    masks = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    for i in range(masks.shape[0]):
        masks[i] = slope_mask_10batch(np.squeeze(data[:,i,:,:]))

    os.makedirs('masks',exist_ok=True)
    for idx, img in enumerate(masks):
        cv2.imwrite(f'masks/mask_{idx}.PNG',(min_max(img)*((2**8)-1)).astype(np.uint8))

    # data_name = str(sys.argv[2])
    # batch = int(sys.argv[1])

    # tot_num_frames = num_frames(data_name)

    # if tot_num_frames-batch<120:
    #     arr = np.zeros((tot_num_frames-batch,500,1500,454),dtype=np.float32)
    # else:
    #     arr = np.zeros((120,500,1500,454),dtype=np.float32)
    
    # j=0
    # for num in range(batch,batch+120):
    #     arr[j] = load_data(data_name,num*2)
    #     if j==arr.shape[0]-1:
    #         break
    #     else:
    #         j+=1
        
    # arr = arr.astype(np.float32)

    # mask_each_batch = np.zeros((arr.shape[0],arr.shape[2],arr.shape[3]), dtype=np.float32)
    # X_shape = (arr.shape[0], arr.shape[1], arr.shape[2],arr.shape[3]) # 120x40x1500x454
    # X = multiprocessing.Array('f', X_shape[0] * X_shape[1] * X_shape[2] * X_shape[3], lock=False)

    # X_np = np.frombuffer(X, dtype=np.float32).reshape(X_shape)
    # np.copyto(X_np, arr)
    # del arr

    # with multiprocessing.Pool(processes=120) as pool:
    #     results = [pool.apply_async(image, args=(x_num, X_shape, X_np[x_num])) for x_num in range(X_shape[0])]
    #     pool.close()
    #     pool.join()

    # for r in results:
    #     m,x_idx = r.get()
    #     mask_each_batch[x_idx] = m


    # with open(f'tmp/{data_name}_mask_{batch}.pickle', 'wb') as handle:
    #     pickle.dump(mask_each_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

