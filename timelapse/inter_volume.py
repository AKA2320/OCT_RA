import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import skimage as ski
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
import cv2
import scipy
from natsort import natsorted
# from sklearn.mixture import GaussianMixture
from skimage.registration import phase_cross_correlation
from scipy import ndimage as scp
from tqdm import tqdm
from skimage.metrics import normalized_root_mse as nrm
# from statsmodels.tsa.stattools import acf
import pickle
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft
import time
import math
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist
# from skimage.feature import SIFT, match_descriptors,plot_matches
# from skimage.feature import ORB
import ants.registration as ants_register
import ants
from scipy.optimize import minimize as minz
from scipy import optimize
import pickle
from skimage.filters import threshold_otsu
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import mean_squared_error as mse
from tifffile import imread as tiffread
import sys
# from util_funcs import *

def load_nested_data_pickle(path):
    pic_paths = []
    for scan_num in os.listdir(path):
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

def y_motion_correction(path):
    data = load_nested_data_pickle(path)
    # n = data.shape[0]
    nn = [np.argmax(np.sum(data[i][0,:,:],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j]  = warp(data[i][j],AffineTransform(matrix=tf_all_nn[i]),order=3)
    return data


if __name__ == '__main__':
    path1 = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_without_H2O2_12_20_2024/registered_cropped_bottom'
    # path2 = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top'

    data_reg = y_motion_correction(path1)
    folder_path = 'inter_volume_registered/Timelapse_without_H2O2/registered_cropped_bottom'
    os.makedirs(folder_path, exist_ok=True)
    for i in tqdm(range(data_reg.shape[0])):
        name = f'scan{i+1}'
        os.makedirs(f'{folder_path}/{name}/', exist_ok=True)
        with open(f'{folder_path}/{name}/{name}.pickle', 'wb') as handle:
            pickle.dump(data_reg[i].astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)