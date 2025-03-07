import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import skimage as ski
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
import cv2
import scipy
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

def mse_fun_tran(shif,x,y):
    # tform = AffineTransform(translation=(shif[0],0))
    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)) ,order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)) ,order=3)

    # cropped_x,cropped_y = crop_img(shif[0],warped_x_stat,warped_y_mov)
    return 2-nmi(warped_x_stat ,warped_y_mov,bins=500)

def reg(scan_num):
    path = f'{scan_num}/'
    pic_paths = os.listdir(path)
    # for i in os.listdir(path):
    #     if i.endswith('.hdf5'):
    #         pic_paths.append(i)
    with h5py.File(path+pic_paths[0], 'r') as hf:
        original_data = hf['volume'][:,200:600,:]

    mid = find_mid(original_data)
    n = original_data.shape[1]

    # finding the bright points in all images in standard interference
    nn = [np.argmax(np.sum(original_data[i][:n//2],axis=1)) for i in range(original_data.shape[0])]
    # intial correcting the y-motion
    tf_all_nn = np.tile(np.eye(3),(original_data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in tqdm(range(original_data.shape[0]),desc='warping'):
        original_data[i][:mid]  = warp(original_data[i][:mid],AffineTransform(matrix=tf_all_nn[i]),order=3)
    # finding the bright points in all images to crop the standard interference
    nn = [np.argmax(np.sum(original_data[i][:n//2],axis=1)) for i in range(original_data.shape[0])]
    UP, DOWN = np.min(nn)-80,np.max(nn)+80
    UP = UP if UP>0 else 0
    DOWN = DOWN if DOWN<original_data.shape[1] else original_data.shape[1]
    # better correcting the y-motion using functions
    tr_all = ants_all_trans(original_data,UP,DOWN) # fucntion definition in util_funcs.py
    for i in tqdm(range(original_data.shape[0]),desc='warping'):
        original_data[i][:mid]  = warp(original_data[i][:mid],AffineTransform(matrix=tr_all[i]),order=3)
    y_corrected_data = scp.median_filter(original_data,size=3)



    UP,DOWN = 20,55
    mir_UP,mir_DOWN = 0,1
    transforms_all = np.tile(np.eye(3),(y_corrected_data.shape[0],1,1))
    for i in tqdm(range(3,y_corrected_data.shape[0]-1)):
        temp_tform_manual = AffineTransform(translation=(0,0))
        static = (min_max((y_corrected_data[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]]))).copy()
        moving = (min_max((y_corrected_data[i+1][np.r_[UP:DOWN,mir_UP:mir_DOWN]]))).copy()
        moving = match_histograms(moving,static)
        for __ in range(5):
            move = minz(method='powell',fun = mse_fun_tran,x0 =(0),bounds = ([(-3,3)]),
                    args = (static
                            ,moving))['x']
            if abs(move[0])<2:
                temp_transform = AffineTransform(translation=(move[0]*2,0))
                static = warp(static, AffineTransform(translation=(-move[0],0)),order=3)
                moving = warp(moving, AffineTransform(translation=(move[0],0)),order=3)
                temp_tform_manual = np.dot(temp_tform_manual,temp_transform)
        transforms_all[i+1:] = np.dot(transforms_all[i+1:],temp_tform_manual)

    for i in tqdm(range(y_corrected_data.shape[0])):
        y_corrected_data[i] = warp(y_corrected_data[i],AffineTransform(matrix=transforms_all[i]),order=3)

    os.makedirs(f'registered/{scan_num}',exist_ok=True)
    hdf5_filename = f'registered/{scan_num}/{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=y_corrected_data, compression='gzip')


if __name__ == "__main__":
    # scan_num = 'scan1'
    done_scans = set([i for i in os.listdir('registered') if (i.startswith('scan'))])
    print(done_scans)
    scans = [i for i in os.listdir() if (i.startswith('scan')) and (i not in done_scans)]
    print('REMAINING',scans)
    for sc in scans:
        print(f'Processing {sc}-----------------------------------------')
        reg(sc)
        print(f'Done Processing {sc}------------------------------------')