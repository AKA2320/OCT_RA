import numpy as np
import pickle
from sklearn.cluster import KMeans,MiniBatchKMeans
import pydicom as dicom
import matplotlib.pylab as plt
import os
import cv2
from natsort import natsorted
from scipy import ndimage as scp
from tqdm import tqdm
import time
import sys
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist
from utils import *
import config
from config import WithH2O2_bottom,WithoutH2O2_bottom
import inspect

def global_min_max(data1,minn,maxx):
    if maxx==0:
        return data1
    else:
        data1 = (data1-minn)/(maxx-minn)
        return data1

def get_rgb(mask,img,percent = 99.9):
    h = (mask*180).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image


if __name__ == '__main__':
    # config_paths = [(name,obj) for name,obj in inspect.getmembers(config) if inspect.isclass(obj)]
    config_paths = [('WithH2O2_bottom',WithH2O2_bottom),('WithoutH2O2_bottom',WithoutH2O2_bottom)]
    loaded_mask = []
    mask_names = []
    for path_name,path_obj in config_paths:
        loaded_mask.append(pickle.load(open(path_obj.mask_save_file_pickle, 'rb')))
        mask_names.append(path_name)

    print('Data Loaded')
    
    # all_values = np.concatenate([arr.flatten() for arr in loaded_mask])
    # global_min = np.min(all_values)
    # global_max = np.max(all_values)
    # del all_values

    # for i in range(len(loaded_mask)):
    #     loaded_mask[i] = global_min_max(loaded_mask[i],global_min,global_max)

    mask_flat = {}
    mask_shapes = {}
    for i in range(len(mask_names)):
        mask_flat[mask_names[i]] = loaded_mask[i].flatten().reshape(-1,1)
        mask_shapes[mask_names[i]] = loaded_mask[i].shape
    del loaded_mask

    print('KMeans Training')
    all_values = np.concatenate([arr for arr in mask_flat.values()]).reshape(-1,1)
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(all_values)
    clust_center = sorted(kmeans.cluster_centers_)
    del kmeans

    print('KMeans Training with defined centers')
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center,max_iter=2).fit(all_values)
    del all_values

    print('KMeans Predicting')
    lbls={}
    for mask_names in list(mask_flat.keys()):
        # kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(mask_flat[mask_names])
        # clust_center = sorted(kmeans.cluster_centers_)
        # lbls[mask_names] = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat[mask_names]).reshape(mask_shapes[mask_names])
        lbls[mask_names] = kmeans.predict(mask_flat[mask_names]).reshape(mask_shapes[mask_names])
        lbls[mask_names] = np.array(lbls[mask_names])/4
    del mask_flat

    print('Saving')
    data_avg = []
    # data_avg_names = []
    for path_name,path_obj in config_paths:
        data_avg = min_max(pickle.load(open(path_obj.avg_data_save_pickle, 'rb')))
        # data_avg_names.append(paths)

        all_rgb_data = np.zeros((data_avg.shape[0],data_avg.shape[1],data_avg.shape[2], data_avg.shape[3],3),dtype=np.uint8)
        for batch_num in tqdm(range(data_avg.shape[0])):
            for slice_num in range(data_avg.shape[1]):
                all_rgb_data[batch_num,slice_num] = (cv2.cvtColor(get_rgb(lbls[path_name][batch_num,slice_num],data_avg[batch_num,slice_num]).astype(np.float32), cv2.COLOR_RGB2BGR)*255).astype(np.uint8)

        with open(path_obj.rgb_save_file_pickle, 'wb') as handle:
            pickle.dump(all_rgb_data.astype(np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
