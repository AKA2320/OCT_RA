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

def x_shift_func(shif, x, y , past_shift):
    x = scp.shift(x, -past_shift,order=3,mode='nearest')
    y = scp.shift(y, past_shift,order=3,mode='nearest')

    warped_x_stat = scp.shift(x, -shif[0],order=3,mode='nearest')
    warped_y_mov = scp.shift(y, shif[0],order=3,mode='nearest')

    return (1-ncc1d(warped_x_stat ,warped_y_mov))

def calc_x_trans(data):
    mid_frame = data.shape[0]//2
    sf_all = []
    for i in tqdm(range(data.shape[0])):
        sf_row = []
        for j in range(data[0].shape[0]):
            ps = 0
            row_st = data[mid_frame][:,250:350][j].copy()
            row_mv = data[i][:,250:350][j].copy()
            for _ in range(5):
                result = minz(method='L-BFGS-B', fun=x_shift_func, x0=0, 
                            bounds= [(-3, 3)], args=(row_st, row_mv, ps))
                ps += result['x']
            sf_row.append(ps * 2)
        sf_all.append(np.array(sf_row))
    return np.squeeze(sf_all)


def load_h5_data(path_scan):
    path = path_scan
    with h5py.File(path, 'r') as hf:
        data = np.array(hf['volume'])
    first_frame = data[10]
    m = data[:,:,10]
    enf_idx = np.argmax(np.sum(m,axis=0))
    enface = data[:,enf_idx,:]
    return first_frame, enface

def apply_transform_data(path_scan , tf, scan_num, data_name_idx):
    path = path_scan
    with h5py.File(path, 'r') as hf:
        data = np.array(hf['volume'])
        for i in range(data.shape[0]):
            data[i] = warp(data[i], AffineTransform(translation=(-tf[i],0)),order=3,mode='edge')

    save_path = f'intervolume_registered/{data_name[data_name_idx]}/{sc}'
    os.makedirs(save_path,exist_ok=True)
    with h5py.File(save_path+f'/{sc}.h5', 'w') as hf:
        hf.create_dataset('volume', data=data.astype(np.float64), compression='gzip',compression_opts=5)
    return None

if __name__ == "__main__":
    # if os.path.exists(f'intervolume_registered/{data_name[0]}'):
    #     done_scans = set([i for i in os.listdir(f'intervolume_registered/{data_name[0]}') if (i.startswith('scan'))])
    #     print(done_scans)
    # else:
    #     done_scans={}
    scans = [i for i in os.listdir(f'intervolume_registered/{data_name[0]}') if (i.startswith('scan'))]
    scans = natsorted(scans)
    print('REMAINING',scans)
    enfaces = []
    for sc in scans:
        print(f'Loading {sc}-----------------------------------------')
        _,data_frame = load_h5_data(f'intervolume_registered/{data_name[0]}/{sc}/{sc}.h5')
        enfaces.append(data_frame)
        # print(f'Done {sc}------------------------------------')
    enfaces = np.array(enfaces)
    x_transform = calc_x_trans(enfaces)
    for idx, sc in enumerate(scans):
        print(f'Processing {sc}-----------------------------------------')
        apply_transform_data(f'intervolume_registered/{data_name[0]}/{sc}/{sc}.h5',x_transform[idx], sc, 0)
        apply_transform_data(f'intervolume_registered/{data_name[1]}/{sc}/{sc}.h5',x_transform[idx], sc, 1)