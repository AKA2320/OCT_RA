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

def shift_func(shif, x, y , past_shift):
    x = scp.shift(x, -past_shift,order=3,mode='nearest')
    y = scp.shift(y, past_shift,order=3,mode='nearest')

    warped_x_stat = scp.shift(x, -shif[0],order=3,mode='nearest')
    warped_y_mov = scp.shift(y, shif[0],order=3,mode='nearest')

    return (1-ncc1d(warped_x_stat ,warped_y_mov))

# def denoise_signal(errs , rows = 10):
#     kk = fft(errs)
#     kk[rows:] = 0
#     kk = abs(ifft(kk))
#     return kk

def run_scans(scan_num):
    path = f'{scan_num}/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.h5'):
            pic_paths.append(i)
    with h5py.File(path+pic_paths[0], 'r') as hf:
        original_data = np.array(hf['volume'])

    original_data = original_data[:,200:,:]

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
    print(UP,DOWN)

    # better correcting the y-motion using functions
    tr_all = ants_all_trans(original_data,UP,DOWN) # fucntion definition in util_funcs.py
    for i in tqdm(range(original_data.shape[0]),desc='warping'):
        original_data[i][:mid]  = warp(original_data[i][:mid],AffineTransform(matrix=tr_all[i]),order=3)

    temp_img = original_data[:,972,250:450].copy()

    kk = []
    for i in range(100):
        move = minz(method='powell',fun = shift_func,x0 =(0),bounds = ([(-100,100)]),
                            args = (temp_img[i]
                                    ,temp_img[i+1]
                                    ,0))['x']
        kk.append(move[0]*2)

    init_move = np.median(kk[::2])
    for i in range(1,temp_img.shape[0],2):
        temp_img[i] = scp.shift(temp_img[i], init_move,order=3,mode='nearest')

    if init_move<0:
        temp_img = temp_img[:,:-int(init_move)]
    else:
        temp_img = temp_img[:,int(init_move):]

    sf = [0]
    for i in tqdm(range(temp_img.shape[0]-1)):
            st = (temp_img[i])
            mv = (temp_img[i+1])
            rt = 0
            past_shift = 0
            for _ in range(15):
                    move = minz(method='L-BFGS-B',fun = shift_func,x0 =(0),bounds = ([(-4,4)]),
                            args = (st
                                    ,mv
                                    ,past_shift))['x']

                    past_shift += move[0]/2
                    rt+=move[0]/2
            sf.append(rt*2)

    # sf = np.array(sf)
    # with open('../dead_mice/shift_IRcard2.pickle', 'wb') as file:
    #     pickle.dump(sf, file)
    # quit()

    # for i in range(1,sf.shape[0]):
    #     sf[i] += sf[i-1]

    for i in tqdm(range(1,len(sf),2)):
        original_data[i] = scp.shift(original_data[i],shift = (0,sf[i]+init_move),order=3,mode='nearest')

    # os.makedirs(f'registered/{scan_num}',exist_ok=True)
    # for i,j in tqdm(enumerate(original_data)):
    #     cv2.imwrite(f'registered/{scan_num}/'+f'frame_test{i}.PNG',j)
    nn = [np.argmax(np.sum(original_data[i][:n//2],axis=1)) for i in range(original_data.shape[0])]
    standard_inter_UP = max(0, np.min(nn) - 50)
    standard_inter_DOWN = standard_inter_UP + 100  # Ensuring difference is always 100
    if standard_inter_DOWN > original_data.shape[1]:
        standard_inter_DOWN = original_data.shape[1]
        standard_inter_UP = max(0, standard_inter_DOWN - 100)  # Adjust UP if DOWN hits boundary


    nn = [np.argmax(np.sum(original_data[i][n//2:1000],axis=1)) for i in range(original_data.shape[0])]
    self_inter_UP = max(0, np.min(nn) - 40)
    self_inter_DOWN = self_inter_UP + 80  # Ensuring difference is always 80
    if self_inter_DOWN > (1000 - n//2):
        self_inter_DOWN = 1000 - n//2
        self_inter_UP = max(0, self_inter_DOWN - 80)  # Adjust UP if DOWN hits boundary

    self_inter_UP = self_inter_UP + n//2
    self_inter_DOWN = self_inter_DOWN + n//2

    # original_data = original_data[:,np.r_[standard_inter_UP:standard_inter_DOWN,self_inter_UP:self_inter_DOWN],:]
    standard_volume = original_data[:,standard_inter_UP:standard_inter_DOWN,:]
    self_volume = original_data[:,self_inter_UP:self_inter_DOWN,:]

    
    
    os.makedirs(f'registered/standard_inter/{scan_num}',exist_ok=True)
    hdf5_filename = f'registered/standard_inter/{scan_num}/{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=standard_volume.astype(np.float64), compression='gzip',compression_opts=5)

    os.makedirs(f'registered/self_inter/{scan_num}',exist_ok=True)
    hdf5_filename = f'registered/self_inter/{scan_num}/{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=self_volume.astype(np.float64), compression='gzip',compression_opts=5)


if __name__ == "__main__":
    if os.path.exists('registered/self_inter'):
        done_scans = set([i for i in os.listdir('registered/self_inter') if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    scans = [i for i in os.listdir() if (i.startswith('scan')) and (i not in done_scans)]
    # scans = ['scan2']
    scans = natsorted(scans)
    print('REMAINING',scans)
    for sc in scans:
        print(f'Processing {sc}-----------------------------------------')
        st = time.perf_counter()
        run_scans(sc)
        end = time.perf_counter()-st
        print(f"Elapsed time: {end:.4f} seconds")
        print(f'Done Processing {sc}------------------------------------')


