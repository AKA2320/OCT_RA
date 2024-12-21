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
# from itertools import permutations 
from skimage.filters import threshold_otsu
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import mean_squared_error as mse
from tifffile import imread as tiffread
import sys
from util_funcs import *


def run_scans(scan_num):
    path = f'{scan_num}/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = np.array(natsorted(pic_paths))
    fst = dicom.dcmread(path+pic_paths[0]).pixel_array

    pics_without_line = np.empty((len(pic_paths),fst.shape[0],fst.shape[1]))
    for i,j in tqdm(enumerate(pic_paths)):
        pics_without_line[i] = dicom.dcmread(path+j).pixel_array
    pics_without_line = pics_without_line.astype(np.float32)


    # Y-MOTION
    mid = find_mid(pics_without_line)
    n = pics_without_line.shape[1]
    nn = [np.argmax(np.sum(pics_without_line[i][:n//2],axis=1)) for i in range(pics_without_line.shape[0])]
    # for i in range(pics_without_line.shape[0]):
    #     nn.append(np.argmax(np.sum(pics_without_line[i][:n//2],axis=1)))

    tf_all_nn = np.tile(np.eye(3),(pics_without_line.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in tqdm(range(pics_without_line.shape[0]),desc='warping'):
        pics_without_line[i][:mid]  = warp(pics_without_line[i][:mid],AffineTransform(matrix=tf_all_nn[i]),order=3)

    nn = [np.argmax(np.sum(pics_without_line[i][:n//2],axis=1)) for i in range(pics_without_line.shape[0])]
    UP, DOWN = np.min(nn)-30,np.max(nn)+30

    tr_all = ants_all_trans(pics_without_line,UP,DOWN)
    for i in tqdm(range(pics_without_line.shape[0]),desc='warping'):
        pics_without_line[i][:mid]  = warp(pics_without_line[i][:mid],AffineTransform(matrix=tr_all[i]),order=3)


    # X-MOTION
    gg = pics_without_line
    UP,DOWN,mir_UP,mir_DOWN = bottom_extract(gg,mid)
    if np.abs(DOWN-UP)<50:
        UP,DOWN,mir_UP,mir_DOWN = top_extract(gg,mid)

    transforms_all = np.tile(np.eye(3),(500,1,1))

    errors_ncc = []
    for i in tqdm(range(gg.shape[0]-1)):
        # mat = scipy.io.loadmat(ants_reg_mapping(min_max(denoise_fft(np.vstack((min_max(gg[i][UP:DOWN]),min_max(gg[i][mir_UP:mir_DOWN])))))
        #                                         ,min_max(denoise_fft(np.vstack((min_max(gg[i+1][UP:DOWN]),min_max(gg[i+1][mir_UP:mir_DOWN]))))))[0])

        mat = scipy.io.loadmat(ants_reg_mapping(min_max(denoise_fft(gg[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]]))
                                                ,min_max(denoise_fft(gg[i+1][np.r_[UP:DOWN,mir_UP:mir_DOWN]])))[0])
        tff = AffineTransform(translation = (mat['AffineTransform_float_2_2'][-2:][1,0],0))
        # transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)
        transforms_all[i+1] = np.dot(transforms_all[i+1],tff)
        temp_transformed_gg = warp(gg[i+1],tff,order=3)
        x_zero_offset, y_zero_offset = non_zero_crop(gg[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]]
                                                    ,temp_transformed_gg[np.r_[UP:DOWN,mir_UP:mir_DOWN]])
        temp_err = (1-ncc(min_max(denoise_fft(np.vstack((min_max(gg[i][UP:DOWN][:,x_zero_offset:y_zero_offset])
                                                        ,min_max(gg[i][mir_UP:mir_DOWN][:,x_zero_offset:y_zero_offset])))))
                        ,min_max(denoise_fft(np.vstack((min_max(temp_transformed_gg[UP:DOWN][:,x_zero_offset:y_zero_offset])
                                                        ,min_max(temp_transformed_gg[mir_UP:mir_DOWN][:,x_zero_offset:y_zero_offset])))))))
        errors_ncc.append(temp_err[0])
    smooth_errors = denoise_signal(errors_ncc[5:])
    peaks = find_peaks(smooth_errors,width=25)[0]
    print('PEAKS ARE HERE NOT PRINTED')
    print(peaks)
    print(transforms_all[:,0,2])

    for frame in peaks:
        gg[frame-2:frame+2].fill(0)
        transforms_all[frame-2:frame+2] = np.eye(3)
    
    for i in range(5,transforms_all.shape[0]):
        transforms_all[i+1:] = np.dot(transforms_all[i+1:],transforms_all[i])


    for i in tqdm(range(gg.shape[0])):
        gg[i] = warp(gg[i],AffineTransform(matrix=transforms_all[i]),order=3)


    # SAVING
    os.makedirs(f'registered/{scan_num}',exist_ok=True)
    for i,j in tqdm(enumerate(gg)):
        cv2.imwrite(f'registered/{scan_num}/'+f'frame_test{i}.PNG',(min_max(j)*((2**16)-1)).astype(np.uint16))

if __name__ == '__main__':
    scans = [i for i in os.listdir() if i.startswith('scan')]
    # for sc in scans:
    #     run_scans(sc)
    #     print(f'------- {sc} ----- DONE -----------------')
    run_scans('scan17')