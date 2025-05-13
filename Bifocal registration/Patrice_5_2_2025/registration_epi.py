from pydicom import dcmread
import matplotlib.pylab as plt
import numpy as np
import os
# import skimage as ski
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
# import cv2
# import scipy

from natsort import natsorted
# from skimage.exposure import match_histograms
# from sklearn.mixture import GaussianMixture
# from skimage.registration import phase_cross_correlation
from scipy.signal import find_peaks
from scipy import ndimage as scp
from tqdm import tqdm
# from skimage.metrics import normalized_root_mse as nrm
# from statsmodels.tsa.stattools import acf
import pickle
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft, ifftshift
import time
# import math
# from skimage.exposure import equalize_hist
# from skimage.exposure import equalize_adapthist
# from skimage.feature import SIFT, match_descriptors,plot_matches
# from skimage.feature import ORB
import ants.registration as ants_register
import ants
from scipy.optimize import minimize as minz
# from scipy.optimize import dual_annealing,fmin_powell
from scipy import optimize
import pickle
# from skimage.filters import threshold_otsu
# from skimage.metrics import normalized_mutual_information as nmi
# from skimage.metrics import mean_squared_error as mse
# from tifffile import imread as tiffread
import sys
from util_funcs import *
import h5py
from ultralytics import YOLO
from collections import defaultdict
import gc

def mse_fun_tran_flat(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=1)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=1)

    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=1)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=1)

    return (1-ncc(warped_x_stat ,warped_y_mov))
    
def ants_all_tran_flat(data,UP_flat,DOWN_flat,static_flat):
    transforms_all = np.tile(np.eye(3),(data.shape[2],1,1))
    for i in tqdm(range(data.shape[2]),desc='tr_all flat'):
        stat = data[:,UP_flat:DOWN_flat,static_flat][::20].copy()
        temp_img = data[:,UP_flat:DOWN_flat,i][::20].copy()

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

def flatten_data(data,UP_flat,DOWN_flat,top_surf):
    static_flat = np.argmax(np.sum(data[:,UP_flat:DOWN_flat,:],axis=(0,1)))
    # finding the bright points in all images in standard interference
    temp_rotated_data = data[:,UP_flat:DOWN_flat,:].transpose(2,1,0)
    nn = [np.argmax(np.sum(temp_rotated_data[i],axis=1)) for i in range(temp_rotated_data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(temp_rotated_data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(-(nn[0]-nn[i]),0)))
    if top_surf:
        for i in tqdm(range(data.shape[2]),desc='warping'):
            data[:,:DOWN_flat,i]  = warp(data[:,:DOWN_flat,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    else:
        for i in tqdm(range(data.shape[2]),desc='warping'):
            data[:,UP_flat:,i]  = warp(data[:,UP_flat:,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)

    tr_all = ants_all_tran_flat(data,UP_flat,DOWN_flat,static_flat)
    if top_surf:
        for i in tqdm(range(data.shape[2]),desc='warping'):
            data[:,:DOWN_flat,i]  = warp(data[:,:DOWN_flat,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    else:
        for i in tqdm(range(data.shape[2]),desc='warping'):
            data[:,UP_flat:,i]  = warp(data[:,UP_flat:,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    return data


def mse_fun_tran_y(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3)
    y = warp(y, AffineTransform(translation=(0,past_shift)),order=3)

    warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3)

    return (1-ncc(warped_x_stat ,warped_y_mov))
    
def ants_all_trans_y(data,UP_y,DOWN_y,static_y_motion):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='tr_all y-motion'):
        stat = data[static_y_motion][UP_y:DOWN_y][:,::20].copy()
        temp_img = data[i][UP_y:DOWN_y][:,::20].copy()
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

def y_motion_correcting(data,UP_y,DOWN_y,top_surf):
    static_y_motion = np.argmax(np.sum(data[:,UP_y:DOWN_y,:],axis=(1,2)))
    # finding the bright points in all images in standard interference
    nn = [np.argmax(np.sum(data[i][UP_y:DOWN_y],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='warping'):
            data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tf_all_nn[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='warping'):
            data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tf_all_nn[i]),order=3)

    tr_all_y = ants_all_trans_y(data,UP_y,DOWN_y,static_y_motion)
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='warping'):
            data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tr_all_y[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='warping'):
            data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tr_all_y[i]),order=3)
    return data

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

def ants_all_trans_x(data,UP_x,DOWN_x,valid_args,enface_extraction_rows):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(0,data.shape[0]-1,2),desc='tr_all'):
        if i not in valid_args:
            continue
        if len(UP_x) and len(DOWN_x)>1:
            stat = data[i,np.r_[UP_x[0]:DOWN_x[0],UP_x[1]:DOWN_x[1]]].copy()
            temp_manual = data[i+1,np.r_[UP_x[0]:DOWN_x[0],UP_x[1]:DOWN_x[1]]].copy()
        else:
            stat = data[i,UP_x[0]:DOWN_x[0]].copy()
            temp_manual = data[i+1,UP_x[0]:DOWN_x[0]].copy()
        # MANUAL
        temp_tform_manual = AffineTransform(translation=(0,0))
        past_shift = 0
        for _ in range(10):
            move = minz(method='powell',fun = mse_fun_tran_x,x0 =(0), bounds=[(-5,5)],
                        args = (stat
                                ,temp_manual
                                ,past_shift))['x']

            past_shift += move[0]
        cross_section = -(past_shift*2)
        enface_shape = data[:,0,:].shape[1]
        enface_wraps = []
        if len(enface_extraction_rows)>0:
            for enf_idx in range(len(enface_extraction_rows)):
                try:
                    temp_enface_shift = get_line_shift(data[i,enface_extraction_rows[enf_idx]],data[i+1,enface_extraction_rows[enf_idx]],enface_shape)
                except:
                    temp_enface_shift = 0
                enface_wraps.append(temp_enface_shift)
        all_warps = [cross_section,*enface_wraps]
        best_warp = check_multiple_warps(data[i], data[i+1], all_warps)
        temp_tform_manual = AffineTransform(translation=(-(all_warps[best_warp]),0))
        transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
        gc.collect()
    return transforms_all

def load_data(dirname, scan_num):
    path = f'{dirname}/{scan_num}/'
    # path = 'intervolume_registered/self_inter/scan5/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.h5'):
            pic_paths.append(i)
    with h5py.File(path+pic_paths[0], 'r') as hf:
        original_data = hf['volume'][:,100:-100,:]
    return original_data

def main(dirname, scan_num):
    original_data = load_data(dirname,scan_num)
    # MODEL PART
    model = YOLO('/Users/akapatil/Documents/feature_extraction/yolo_feature_extraction/yolov12s_best.pt')
    static_flat = np.argmax(np.sum(original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res = model.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=0)
    result_list = res[0].summary()
    grouped = defaultdict(list)
    for item in result_list:
        grouped[item['name']].append(item)
    filtered_summary = []
    for group in grouped.values():
        top_two = sorted(group, key=lambda x: x['confidence'], reverse=True)[:2]
        filtered_summary.extend(top_two)
    result_list = filtered_summary
    if len(result_list)==0:
        print(f'NO SURFACE DETECTED: {scan_num}')
        return
    if len(result_list)>0:
        surface_coords = []
        for surfaces in result_list:
            surface_coords.append([int(surfaces['box']['y1'])-20,int(surfaces['box']['y2'])+20])
        if len(surface_coords)>1:
            surface_coords = np.sort(surface_coords,axis=0)
    surface_coords = np.squeeze(np.array(surface_coords))
    surface_coords = np.where(surface_coords<0,0,surface_coords)
    
    # sum_img = np.max(original_data[:,:,:],axis=2)
    # peaks = find_peaks(np.sum(sum_img,axis=0),distance = 30)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(sum_img,axis=0)[peaks])[-2:]]
    # enface_extraction_rows = np.sort(enface_extraction_rows)
    # print('ROWS:',enface_extraction_rows)
    
    # FLATTENING PART
    print('SURFACE COORDS:',surface_coords)
    if surface_coords.ndim==1 or surface_coords.shape[0]==1:
        static_flat = np.argmax(np.sum(original_data[:,surface_coords[0]:surface_coords[1],:],axis=(0,1)))
    else:
        static_flat = np.argmax(np.sum(original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    if surface_coords.ndim==1:
        surface_coords = surface_coords.reshape(1,-1)
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_flat,DOWN_flat = surface_coords[i][0], surface_coords[i][1]
        UP_flat = max(UP_flat,0)
        DOWN_flat = min(DOWN_flat, original_data.shape[2])
        original_data = flatten_data(original_data,UP_flat,DOWN_flat,top_surf)
        top_surf = False

    # Y-MOTION PART
    if surface_coords.ndim==1:
        surface_coords = surface_coords.reshape(1,-1)
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_y,DOWN_y = surface_coords[i][0], surface_coords[i][1]
        UP_y = max(UP_y,0)
        DOWN_y = min(DOWN_y, original_data.shape[2])
        original_data = y_motion_correcting(original_data,UP_y,DOWN_y,top_surf)
        top_surf = False

    # X-MOTION PART
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res = model.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=[0,1])
    result_list = res[0].summary()
    grouped = defaultdict(list)
    for item in result_list:
        grouped[item['name']].append(item)
    filtered_summary = []
    for group in grouped.values():
        top_two = sorted(group, key=lambda x: x['confidence'], reverse=True)[:2]
        filtered_summary.extend(top_two)
    result_list = filtered_summary

    if len(result_list)>0:
        surface_coords = []
        cells_coords = []
        for detections in result_list:
            if detections['name'] == 'surface':
                surface_coords.append([int(detections['box']['y1'])-10,int(detections['box']['y2'])+10])
            if detections['name'] == 'cells':
                cells_coords.append([int(detections['box']['y1'])-5,int(detections['box']['y2'])+5])
        if len(surface_coords)>1:
            surface_coords = np.sort(surface_coords,axis=0)
        if len(cells_coords)>1:
            cells_coords = np.sort(cells_coords,axis=0)
    cells_coords = np.squeeze(np.array(cells_coords))
    surface_coords = np.squeeze(np.array(surface_coords))
    cells_coords = np.where(cells_coords<0,0,cells_coords)
    surface_coords = np.where(surface_coords<0,0,surface_coords)

    if cells_coords.size==0 and surface_coords.size==0:
        print(f'NO SURFACE OR CELLS DETECTED: {scan_num}')
        return

    enface_extraction_rows = []
    if surface_coords.size!=0:
        print('SURFACE COORDS:',surface_coords)
        if surface_coords.ndim==1 or surface_coords.shape[0]==1:
            # print('SURFACE COORDS:',surface_coords)
            static_y_motion = np.argmax(np.sum(original_data[:,surface_coords[0]:surface_coords[1],:],axis=(1,2)))    
        else:
            static_y_motion = np.argmax(np.sum(original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(1,2)))    
        errs = []
        for i in range(original_data.shape[0]):
            errs.append(ncc(original_data[static_y_motion,:,:],original_data[i,:,:])[0])
        errs = np.squeeze(errs)
        valid_args = np.squeeze(np.argwhere(errs>0.7))
        if surface_coords.ndim==1:
            surface_coords = surface_coords.reshape(1,-1)
        if surface_coords.shape[0]==1:
            # UP_surf = max(0,surface_coords[0,0])
            # DOWN_surf = min(original_data.shape[2],surface_coords[0,1])
            val = np.argmax(np.sum(np.max(original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=0),axis=1))
            enface_extraction_rows.append(val)
        else:
            for i in range(surface_coords.shape[0]):
                val = np.argmax(np.sum(np.max(original_data[:,surface_coords[i,0]:surface_coords[i,1],:],axis=0),axis=1))
                enface_extraction_rows.append(val)
    else:
        valid_args = np.arange(original_data.shape[0])

    
    if cells_coords.ndim==1:
        cells_coords = cells_coords.reshape(1,-1)
    if cells_coords.shape[0]==1:
        UP_x, DOWN_x = [cells_coords[0][0]], [cells_coords[0][1]]
    else:
        UP_x, DOWN_x = (cells_coords[:,0]), (cells_coords[:,1])

    '''
    # sum_img = np.max(original_data[:,:,:],axis=2)
    # peaks = find_peaks(np.sum(sum_img,axis=0),distance = 30)[0]
    # enface_extraction_rows = peaks[np.argsort(np.sum(sum_img,axis=0)[peaks])[-2:]]
    # enface_extraction_rows = np.sort(enface_extraction_rows)
    # print('ROWS:',enface_extraction_rows)
    # UP_x,DOWN_x = enface_extraction_rows.min()-50, enface_extraction_rows.min()-10
    # UP_x = max(UP_x,0)
    # DOWN_x = min(DOWN_x, original_data.shape[2])

    # static_y_motion = np.argmax(np.sum(original_data[:,UP_x:DOWN_x,:],axis=(1,2)))
    # errs = []
    # for i in range(original_data.shape[0]):
    #     errs.append(ncc(original_data[static_y_motion,:,:],original_data[i,:,:])[0])
    # errs = np.squeeze(errs)
    # valid_args = np.squeeze(np.argwhere(errs>0.7))
    '''

    tr_all = ants_all_trans_x(original_data,UP_x,DOWN_x,valid_args,enface_extraction_rows)
    for i in tqdm(range(1,original_data.shape[0],2),desc='warping'):
        original_data[i]  = warp(original_data[i],AffineTransform(matrix=tr_all[i]),order=3)

    folder_save = 'registered_epi'
    os.makedirs(folder_save,exist_ok=True)
    hdf5_filename = f'{folder_save}/{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=original_data.astype(np.float64), compression='gzip',compression_opts=5)


if __name__ == "__main__":
    data_dirname = 'batch_epi'
    if os.path.exists('registered_epi/'):
        done_scans = set([i for i in os.listdir('registered_epi/') if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    # scans = ['scan2']
    scans = natsorted(scans)
    # scans.remove('scan15')
    print('REMAINING',scans)
    for sc in scans:
        print(f'Processing {sc}-----------------------------------------')
        st = time.perf_counter()
        main(data_dirname,sc)
        end = time.perf_counter()-st
        print(f"Elapsed time: {end:.4f} seconds")
        print(f'Done Processing {sc}------------------------------------')