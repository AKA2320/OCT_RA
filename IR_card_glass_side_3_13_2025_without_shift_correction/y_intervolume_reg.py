import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import skimage as ski
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
import cv2
import scipy
import time
from skimage.filters import unsharp_mask
from natsort import natsorted
from skimage.exposure import match_histograms
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
from scipy.optimize import dual_annealing,fmin_powell
from scipy import optimize
import pickle
from skimage.filters import threshold_otsu
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import mean_squared_error as mse
from tifffile import imread as tiffread
import sys
from util_funcs import *
import h5py

data_name = ['self_inter','standard_inter']

def mse_fun_tran_y(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3,mode='edge')
    y = warp(y, AffineTransform(translation=(0,past_shift)),order=3,mode='edge')

    warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3,mode='edge')
    warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3,mode='edge')

    return (1-ncc(warped_x_stat ,warped_y_mov))

def cacl_y_trans(data):
    y_transform_intervolume = [0]
    for i in tqdm(range(data.shape[0]-1)):
        st = data[i].copy()
        mv = data[i+1].copy()
        rt = 0
        past_shift = 0
        for _ in range(10):
                move = minz(method='L-BFGS-B',fun = mse_fun_tran_y,x0 =(0),bounds = ([(-3,3)]),
                        args = (st
                                ,mv
                                ,past_shift))['x']
                past_shift += move[0]
                rt += move[0]
        y_transform_intervolume.append(rt*2)
    y_transform_intervolume = np.array(y_transform_intervolume)
    for i in range(1,y_transform_intervolume.shape[0]):
        y_transform_intervolume[i] += y_transform_intervolume[i-1]
    return y_transform_intervolume


def load_h5_data(path_scan):
    path = path_scan
    with h5py.File(path, 'r') as hf:
        data = np.array(hf['volume'])
    first_frame = data[10]
    m = data[:,:,10]
    enf_idx = np.argmax(np.sum(m[:,m.shape[1]//2:],axis=0))
    enface = data[:,enf_idx,:]
    return first_frame, enface

def apply_transform_data(path_scan , tf, scan_num,data_name_idx):
    path = path_scan
    with h5py.File(path, 'r') as hf:
        data = np.array(hf['volume'])
        for i in range(data.shape[0]):
            data[i] = warp(data[i], AffineTransform(translation=(0,tf)),order=3,mode='edge')

    save_path = f'intervolume_registered/{data_name[data_name_idx]}/{sc}'
    os.makedirs(save_path,exist_ok=True)
    with h5py.File(save_path+f'/{sc}.h5', 'w') as hf:
        hf.create_dataset('volume', data=data.astype(np.float64), compression='gzip',compression_opts=5)
    return None


if __name__ == "__main__":
    # if os.path.exists(f'intervolume_registered/{data_name}'):
    #     done_scans = set([i for i in os.listdir(f'intervolume_registered/{data_name}') if (i.startswith('scan'))])
    #     print(done_scans)
    # else:
    #     done_scans={}
    for name_idx in range(len(data_name)):
        done_scans = {}
        scans = [i for i in os.listdir(f'registered/{data_name[name_idx]}') if (i.startswith('scan')) and (i not in done_scans)]
        scans = natsorted(scans)
        print('REMAINING',scans)
        frames = []
        for sc in scans:
            print(f'Loading {sc}-----------------------------------------')
            data_frame, _ = load_h5_data(f'registered/{data_name[name_idx]}/{sc}/{sc}.h5')
            frames.append(data_frame)
            # print(f'Done {sc}------------------------------------')
        frames = np.array(frames)
        y_transform = cacl_y_trans(frames)
        for idx, sc in enumerate(scans):
            print(f'Processing {sc}-----------------------------------------')
            apply_transform_data(f'registered/{data_name[name_idx]}/{sc}/{sc}.h5',y_transform[idx], sc, name_idx)
    