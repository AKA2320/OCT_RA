import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as scp
#from statsmodels.tsa.stattools import acf
from natsort import natsorted
import cv2
from numpy.fft import fft2,fft,ifft
from skimage.transform import warp, AffineTransform
from scipy.optimize import curve_fit

def min_max(data1):
    maxx = np.max(data1)
    if maxx==0:
        return data1
    else:
        data1 = (data1-np.min(data1))/(maxx-np.min(data1))
        return data1

def load_nested_data_pickle(path, num):
    pic_paths = []
    for scan_num in os.listdir(path)[0:num]:
        if scan_num.startswith('scan'):
            pic_paths.append(os.path.join(path,scan_num,f'{scan_num}.pickle'))
    pic_paths = natsorted(pic_paths)
    with open(f'{pic_paths[0]}', 'rb') as handle:
        b = pickle.load(handle)
    data = np.zeros((len(pic_paths),b.shape[0],b.shape[1],b.shape[2]))

    for idx,img_path in enumerate(pic_paths):
        with open(img_path, 'rb') as handle:
            temp = pickle.load(handle)
        data[idx]=(temp.copy())
    data = data.astype(np.float32)
    return data

def load_nested_data_png(path):
    pic_paths = []
    for scan_num in os.listdir(path):
        if scan_num.startswith('scan'):
            pic_paths.append(os.path.join(path,scan_num))
    pic_paths = natsorted(pic_paths)
    print(os.path.join(pic_paths[0],os.listdir(pic_paths[0])[0]))
    # print(pic_paths[0]+os.listdir(pic_paths[0])[0])
    temp_img = cv2.imread(os.path.join(pic_paths[0],os.listdir(pic_paths[0])[0]),cv2.IMREAD_UNCHANGED) 
    data = np.zeros((len(pic_paths),len(os.listdir(pic_paths[0])),temp_img.shape[0],temp_img.shape[1]))

    for main_idx,img_paths in enumerate(pic_paths):
        all_img_paths = natsorted(os.listdir(img_paths))
        for idx,img_path in enumerate(all_img_paths):
            temp = cv2.imread(os.path.join(img_paths,img_path),cv2.IMREAD_UNCHANGED)
            data[main_idx,idx]=(temp.copy())
    data = data.astype(np.float32)
    return data

def load_data_png(path):
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.PNG') or i.endswith('.png'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)

    temp_img = cv2.imread(path+pic_paths[0],cv2.IMREAD_UNCHANGED) 
    imgs_from_folder = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1]))
    # imgs_from_folder = []
    for i,j in enumerate(pic_paths):
        aa = cv2.imread(path+j,cv2.IMREAD_UNCHANGED)
        imgs_from_folder[i] = aa.copy()
    imgs_from_folder = imgs_from_folder.astype(np.float32)
    return imgs_from_folder

def slope_mask(slope_arr):
    mask1 = np.zeros_like(slope_arr[0],dtype=np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=slope_arr,axis=0)
    slope_arr = np.apply_along_axis(func1d=min_max,arr=slope_arr,axis=0)
    for x in range(slope_arr.shape[1]):
        for y in range(slope_arr.shape[2]):
            data1 = slope_arr[:,x,y]
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*(5*std_mask)

def ymotion(data):
    # n = data.shape[0]
    nn = [np.argmax(np.sum(data[i][0,:,:],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j]  = warp(data[i][j],AffineTransform(matrix=tf_all_nn[i]),order=3)
    return data



## aLIV related function
def AVG_LIV(oneD_data, time_index_list):
    log_data = 10 * np.log10(oneD_data + 1e-8)#convert the intensity to log data, add small number to prevent 0 from occuring  
    time_avgLIV = np.zeros((len(time_index_list), 2))#data structure to store time interval and its avgLIV
    for i in range(len(time_index_list)):
        index_data_group = time_index_list[i][1]
        num_data_group = len(index_data_group)#number of data groups that belong to this specific time interval
        LIV_group = np.zeros(num_data_group)
        for j in range(num_data_group):
            sub_log_data = log_data[index_data_group[j]]
            avglog = np.sum(sub_log_data) / (np.max(index_data_group[j])-np.min(index_data_group[j]))#time average of LIV
            sub_log_substraction = sub_log_data - avglog
            sub_LIV = np.mean(np.square(sub_log_substraction))#LIV of this particular sub dataset
            LIV_group[j] = sub_LIV
        time_avgLIV[i, :] = [time_index_list[i][0], np.mean(LIV_group)]#the average LIV of this particular time interval
    return time_avgLIV


def LIV_fun(Tw, a, tau):
    return a * (1 - np.exp(-Tw / tau))

#function in multiprocessing

def LIV_multiprocess(shared_array, shape, i, j, k, time_lapse_string, time_index_list):
    array_2d = np.frombuffer(shared_array.get_obj()).reshape(shape)
    Average_LIV = AVG_LIV(time_lapse_string, time_index_list)
    Average_LIV[~np.isfinite(Average_LIV)] = np.mean(Average_LIV[np.isfinite(Average_LIV)])#remove infinity and NaN
    popt = curve_fit(LIV_fun, Average_LIV[:, 0], Average_LIV[:, 1], bounds=(0, np.inf))[0]
    array_2d[i, j, k] = 1 / popt[1]