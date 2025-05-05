from pydicom import dcmread
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
from scipy.signal import find_peaks
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



def load_data(scan_num):
    path = f'batch/{scan_num}/'
    # path = 'intervolume_registered/self_inter/scan5/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.h5'):
            pic_paths.append(i)
    with h5py.File(path+pic_paths[0], 'r') as hf:
        original_data = np.array(hf['volume'][:,400:550,:])
    return original_data


def mse_fun_tran_flat(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=1)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=1)

    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=1)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=1)

    return (1-ncc(warped_x_stat ,warped_y_mov))
    
def ants_all_tran_flat(data,UP,DOWN, static_flat):
    transforms_all = np.tile(np.eye(3),(data.shape[2],1,1))
    for i in tqdm(range(data.shape[2]),desc='tr_all'):
        stat = data[:,UP:DOWN,static_flat][::20].copy()
        temp_img = data[:,UP:DOWN,i][::20].copy()

        # MANUAL
        # temp_tform_manual = AffineTransform(translation=(0,0))
        past_shift = 0
        for _ in range(10):
            move = minz(method='powell',fun = mse_fun_tran_flat,x0 =(0), bounds=[(-3,3)],
                        args = (stat
                                ,temp_img
                                ,past_shift))['x']

            past_shift += move[0]
        temp_tform_manual = AffineTransform(translation=(past_shift*2,0))
        transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all


def mse_fun_tran_y(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3)
    y = warp(y, AffineTransform(translation=(0,past_shift)),order=3)

    warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3)

    return (1-ncc(warped_x_stat ,warped_y_mov))
    
def ants_all_trans_y(data,UP,DOWN,static_y_motion):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='tr_all'):
        stat = data[static_y_motion][UP:DOWN][:,::20].copy()
        temp_img = data[i][UP:DOWN][:,::20].copy()
        # MANUAL
        # temp_tform_manual = AffineTransform(translation=(0,0))
        past_shift = 0
        for _ in range(10):
            move = minz(method='powell',fun = mse_fun_tran_y,x0 =(0), bounds=[(-2,2)],
                        args = (stat
                                ,temp_img
                                ,past_shift))['x']

            past_shift += move[0]
        temp_tform_manual = AffineTransform(matrix = AffineTransform(translation=(0,past_shift*2)))
        transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def shift_func(shif, x, y , past_shift):
    x = scp.shift(x, -past_shift,order=3,mode='nearest')
    y = scp.shift(y, past_shift,order=3,mode='nearest')

    warped_x_stat = scp.shift(x, -shif[0],order=3,mode='nearest')
    warped_y_mov = scp.shift(y, shif[0],order=3,mode='nearest')

    return (1-ncc1d(warped_x_stat ,warped_y_mov))

def ncc1d(array1, array2):
    correlation = np.correlate(array1, array2, mode='valid')
    array1_norm = np.linalg.norm(array1)
    array2_norm = np.linalg.norm(array2)
    if array1_norm == 0 or array2_norm == 0:
        return np.zeros_like(correlation)
    normalized_correlation = correlation / (array1_norm * array2_norm)
    return normalized_correlation

def mse_fun_tran_x(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=3)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=3)

    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=3)

    return (1-ncc(warped_x_stat ,warped_y_mov))

def get_line_shift(line_1d_stat, line_1d_mov,enface_shape):
    # grad_feat = np.argmax(np.abs(np.gradient(line_1d_stat)[5:-5]))+5
    # # print(grad_feat,i)
    # grad_feat = max(20,grad_feat)
    # grad_feat = min(grad_feat,enface_shape-20)
    # st = line_1d_stat[grad_feat-20:grad_feat+20]
    # mv = line_1d_mov[grad_feat-20:grad_feat+20]
    st = line_1d_stat
    mv = line_1d_mov
    past_shift = 0
    for _ in range(10):
        move = minz(method='powell',fun = shift_func,x0 = (0),bounds =[(-5,5)],
                args = (st
                        ,mv
                        ,past_shift))['x']
        past_shift += move[0]
    return past_shift*2

def check_best_warp(stat, mov, value, is_shift_value = False):
    # if is_shift_value:
    err = ncc(stat,warp(mov, AffineTransform(translation=(-value,0)),order=3))
    return err
    # err = ncc(stat,warp(mov, AffineTransform(matrix=value),order=3))
    # return err

def check_multiple_warps(stat_img, mov_img, *args):
    errors = []
    warps = args[0]
    # errors.append(check_best_warp(stat_img, mov_img, warps[0], is_shift_value = False))
    for warp_value in range(len(warps)):
        errors.append(check_best_warp(stat_img, mov_img, warps[warp_value]))
    # print(errors)
    return np.argmax(errors)

import gc

def ants_all_trans_x(data,UP,DOWN,enface_extraction_rows):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(0,data.shape[0]-1,2),desc='tr_all'):
        stat = data[i][UP:DOWN].copy()
        temp_manual = data[i+1][UP:DOWN].copy()
        # MANUAL
        # temp_tform_manual = AffineTransform(translation=(0,0))
        # past_shift = 0
        # for _ in range(10):
        #     move = minz(method='powell',fun = mse_fun_tran_x,x0 =(0), bounds=[(-5,5)],
        #                 args = (stat
        #                         ,temp_manual
        #                         ,past_shift))['x']

        #     past_shift += move[0]
        # cross_section = -(past_shift*2)
        # temp_tform_manual = AffineTransform(matrix = AffineTransform(translation=(past_shift*2,0)))
        # transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        # gc.collect()
        enface_shape = data[:,0,:].shape[1]
        # enface_line_standard = get_line_shift(data[i,enface_extraction_rows[0]],data[i+1,enface_extraction_rows[0]],enface_shape)
        # enface_line_endo = get_line_shift(data[i,800],data[i+1,800],enface_shape)
        enface_line_self = get_line_shift(data[i,enface_extraction_rows],data[i+1,enface_extraction_rows],enface_shape)

        all_warps = [enface_line_self]
        best_warp = check_multiple_warps(data[i], data[i+1], all_warps)

        temp_tform_manual = AffineTransform(translation=(-(all_warps[best_warp]),0))
        transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
        gc.collect()

    return transforms_all


def main(scan_num):
    original_data = load_data(scan_num)
    # sum_img = np.max(original_data[:,:,:],axis=2)
    # peaks = find_peaks(np.sum(sum_img,axis=0),distance = 30)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(sum_img,axis=0)[peaks])[-2:]]
    # # UP,DOWN = enface_extraction_rows.min()-50,enface_extraction_rows.min()+50
    # mid = enface_extraction_rows.max() - 50
    # top_surface_extraction = np.argmax(np.sum(np.max(original_data[:,:mid,:],axis=0),axis=1))
    # UP,DOWN = top_surface_extraction-40, top_surface_extraction+40
    # UP = max(UP,0)
    # DOWN = min(DOWN, original_data.shape[2])
    val = np.argmax(np.sum(np.max(original_data[:,:,:],axis=0),axis=1))
    UP,DOWN = val-30, val+30
    static_flat = np.argmax(np.sum(original_data[:,UP:DOWN,:],axis=(0,1)))

    ####### FLAT
    # n = original_data.shape[1]
    # finding the bright points in all images in standard interference
    # temp_rotated_data = original_data[:,UP:DOWN,:].transpose(2,1,0)
    # nn = [np.argmax(np.sum(temp_rotated_data[i],axis=1)) for i in range(temp_rotated_data.shape[0])]
    # tf_all_nn = np.tile(np.eye(3),(temp_rotated_data.shape[0],1,1))
    # for i in range(tf_all_nn.shape[0]):
    #     tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(-(nn[0]-nn[i]),0)))
    # for i in tqdm(range(original_data.shape[2]),desc='warping'):
    #     original_data[:,UP:DOWN,i]  = warp(original_data[:,UP:DOWN,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    tr_all = ants_all_tran_flat(original_data,UP,DOWN,static_flat)
    for i in tqdm(range(original_data.shape[2]),desc='warping'):
        original_data[:,UP:DOWN,i]  = warp(original_data[:,UP:DOWN,i] ,AffineTransform(matrix=tr_all[i]),order=3)

    ######## Y-MOTION
    # sum_img = np.sum(original_data[:,:,:],axis=2)
    # peaks = find_peaks(np.sum(sum_img,axis=0),distance = 10)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(sum_img,axis=0)[peaks])[-2:]]
    # UP,DOWN = enface_extraction_rows.min()-50,enface_extraction_rows.min()+50
    # UP = max(UP,0)
    # DOWN = min(DOWN, original_data.shape[2])
    static_y_motion = np.argmax(np.sum(original_data[:,UP:DOWN,:],axis=(1,2)))
    # n = original_data.shape[1]
    # finding the bright points in all images in standard interference
    # nn = [np.argmax(np.sum(original_data[i][UP:DOWN],axis=1)) for i in range(original_data.shape[0])]
    # tf_all_nn = np.tile(np.eye(3),(original_data.shape[0],1,1))
    # for i in range(tf_all_nn.shape[0]):
    #     tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    # for i in tqdm(range(original_data.shape[0]),desc='warping'):
    #     original_data[i][UP:DOWN]  = warp(original_data[i][UP:DOWN],AffineTransform(matrix=tf_all_nn[i]),order=3)
    tr_all_y = ants_all_trans_y(original_data,UP,DOWN,static_y_motion)
    for i in tqdm(range(original_data.shape[0]),desc='warping'):
        original_data[i][UP:DOWN]  = warp(original_data[i][UP:DOWN],AffineTransform(matrix=tr_all_y[i]),order=3)
    
    ######## X-MOTION
    # sum_img = np.sum(original_data[:,:,:],axis=2)
    # peaks = find_peaks(np.sum(sum_img,axis=0),distance = 10)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(sum_img,axis=0)[peaks])[-2:]]
    # UP,DOWN = enface_extraction_rows.min()-50,enface_extraction_rows.min()+50
    # UP = max(UP,0)
    # DOWN = min(DOWN, original_data.shape[2])
    # peaks = find_peaks(np.sum(original_data[:,:,static_flat],axis=0),distance = 10)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(original_data[:,:,static_flat],axis=0)[peaks])[-2:]]
    # top_surface = enface_extraction_rows.min()
    # top_surface_extraction = np.argmax(np.sum(np.max(original_data[:,:mid,:],axis=0),axis=1))
    # UP_x,DOWN_x = top_surface_extraction-60,top_surface_extraction-10
    # UP_x = max(UP_x,0)
    # DOWN_x = min(DOWN_x, original_data.shape[2])
    val = np.argmax(np.sum(np.max(original_data[:,:,:],axis=0),axis=1))
    # UP,DOWN = val-30, val+30
    tr_all_x = ants_all_trans_x(original_data,0,10,val)
    for i in tqdm(range(1,original_data.shape[0],2),desc='warping'):
        original_data[i]  = warp(original_data[i],AffineTransform(matrix=tr_all_x[i]),order=3)

    os.makedirs(f'registered/full/',exist_ok=True)
    hdf5_filename = f'registered/full/{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=original_data.astype(np.float64), compression='gzip',compression_opts=5)

if __name__ == "__main__":
    if os.path.exists('registered/full'):
        done_scans = set([i for i in os.listdir('registered/full') if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    scans = [i for i in os.listdir('batch') if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    # scans = ['scan2']
    scans = natsorted(scans)
    print('REMAINING',scans)
    for sc in scans:
        print(f'Processing {sc}-----------------------------------------')
        st = time.perf_counter()
        main(sc)
        end = time.perf_counter()-st
        print(f"Elapsed time: {end:.4f} seconds")
        print(f'Done Processing {sc}------------------------------------')