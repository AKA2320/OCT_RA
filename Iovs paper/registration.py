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
from util_funcs import *


def run_scans(scan_num):

    # lOADING THE DATA
    path = f'{scan_num}/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = np.array(natsorted(pic_paths))
    fst = dicom.dcmread(path+pic_paths[0]).pixel_array

    # Creating empty array for the data
    original_data = np.empty((len(pic_paths),fst.shape[0],fst.shape[1]))
    for i,j in tqdm(enumerate(pic_paths)):
        original_data[i] = dicom.dcmread(path+j).pixel_array

    # Using float32 data
    original_data = original_data.astype(np.float32)


    ###### Y-MOTION

    # find mid algo to find where to divide the data for standard and self interference
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

    print(UP,DOWN)

    # better correcting the y-motion using functions
    tr_all = ants_all_trans(original_data,UP,DOWN) # fucntion definition in util_funcs.py
    for i in tqdm(range(original_data.shape[0]),desc='warping'):
        original_data[i][:mid]  = warp(original_data[i][:mid],AffineTransform(matrix=tr_all[i]),order=3)

    transforms_all = original_data

    ##### X-MOTION without anomaly
    # Extract the cropped region for cells near standard and self interference
    UP,DOWN,mir_UP,mir_DOWN = bottom_extract(y_corrected_data,mid) # bottom_extract found in util_funcs
    if np.abs(DOWN-UP)<50:
        UP,DOWN,mir_UP,mir_DOWN = top_extract(y_corrected_data,mid) # top_extract found in util_funcs

    transforms_all = np.tile(np.eye(3),(y_corrected_data.shape[0],1,1))


    for i in tqdm(range(y_corrected_data.shape[0]-1)):
        mat = scipy.io.loadmat(ants_reg_mapping(min_max(denoise_fft(y_corrected_data[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]])) # denoise_fft found in util_funcs
                                                ,min_max(denoise_fft(y_corrected_data[i+1][np.r_[UP:DOWN,mir_UP:mir_DOWN]])))[0])
        tff = AffineTransform(translation = (mat['AffineTransform_float_2_2'][-2:][1,0],0))
        transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)

    # warp the images
    for i in tqdm(range(y_corrected_data.shape[0])):
        y_corrected_data[i] = warp(y_corrected_data[i],AffineTransform(matrix=transforms_all[i]),order=3)

    '''
    ##### X-MOTION with anomaly removal

    UP,DOWN,mir_UP,mir_DOWN = bottom_extract(y_corrected_data,mid)
    if np.abs(DOWN-UP)<50:
        UP,DOWN,mir_UP,mir_DOWN = top_extract(y_corrected_data,mid)

    transforms_all = np.tile(np.eye(3),(500,1,1))
    errors_ncc = []
    for i in tqdm(range(y_corrected_data.shape[0]-1)):
        mat = scipy.io.loadmat(ants_reg_mapping(min_max(denoise_fft(y_corrected_data[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]]))
                                                ,min_max(denoise_fft(y_corrected_data[i+1][np.r_[UP:DOWN,mir_UP:mir_DOWN]])))[0])
        tff = AffineTransform(translation = (mat['AffineTransform_float_2_2'][-2:][1,0],0))
        # transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)
        transforms_all[i+1] = np.dot(transforms_all[i+1],tff)
        temp_transformed_y_corrected_data = warp(y_corrected_data[i+1],tff,order=3)

        ## USING THE CELLS
        # x_zero_offset, y_zero_offset = non_zero_crop(y_corrected_data[i][np.r_[UP:DOWN,mir_UP:mir_DOWN]]
        #                                             ,temp_transformed_y_corrected_data[np.r_[UP:DOWN,mir_UP:mir_DOWN]])
        # temp_err = (1-ncc(min_max(denoise_fft(np.vstack((min_max(y_corrected_data[i][UP:DOWN][:,x_zero_offset:y_zero_offset])
        #                                                 ,min_max(y_corrected_data[i][mir_UP:mir_DOWN][:,x_zero_offset:y_zero_offset])))))
        #                 ,min_max(denoise_fft(np.vstack((min_max(temp_transformed_y_corrected_data[UP:DOWN][:,x_zero_offset:y_zero_offset])
        #                                                 ,min_max(temp_transformed_y_corrected_data[mir_UP:mir_DOWN][:,x_zero_offset:y_zero_offset])))))))

        ## USING THE SURFACE
        nn = [np.argmax(np.sum(y_corrected_data[i][:n//2],axis=1)) for i in range(y_corrected_data.shape[0])]
        UP_err_calc, DOWN_err_calc = np.min(nn)-30,np.max(nn)+30
        x_zero_offset, y_zero_offset = non_zero_crop(y_corrected_data[i][UP_err_calc:DOWN_err_calc]
                                                    ,temp_transformed_y_corrected_data[UP_err_calc:DOWN_err_calc])
        temp_err = 1-ncc(min_max(y_corrected_data[i][UP_err_calc:DOWN_err_calc]),min_max(temp_transformed_y_corrected_data[UP_err_calc:DOWN_err_calc]))
        
        errors_ncc.append(temp_err[0])
    smooth_errors = denoise_signal1D_err_calc(errors_ncc[:])
    peaks = find_peaks(smooth_errors,width=10,prominence=0.01)[0]
    print('PEAKS ARE HERE PRINTED')
    print(peaks)
    frame_ranges = np.squeeze(np.dstack((np.round(find_peaks(smooth_errors,width=10,prominence=0.01)[1]['left_ips'])
                            ,np.round(find_peaks(smooth_errors,width=10,prominence=0.01)[1]['right_ips'])))).astype(int)
    if frame_ranges.ndim==1:
        frame_ranges = frame_ranges.reshape(1,-1)

    print(frame_ranges,frame_ranges.shape)
    for frame_l,frame_r in frame_ranges:
        y_corrected_data[frame_l:frame_r].fill(0)
        transforms_all[frame_l:frame_r]= np.eye(3)    
    # print(transforms_all[:,0,2])


    transforms_all_corrected = np.tile(np.eye(3),(500,1,1))
    for i in range(transforms_all_corrected.shape[0]):
        transforms_all_corrected[i:] = np.dot(transforms_all_corrected[i:],transforms_all[i])

    # print('PEAKS ARE HERE NOT PRINTED')
    # print(transforms_all_corrected[:,0,2])

    for i in tqdm(range(y_corrected_data.shape[0])):
        y_corrected_data[i] = warp(y_corrected_data[i],AffineTransform(matrix=transforms_all_corrected[i]),order=3)
    '''

    # SAVING
    os.makedirs(f'registered/{scan_num}',exist_ok=True)
    for i,j in tqdm(enumerate(y_corrected_data)):
        cv2.imwrite(f'registered/{scan_num}/'+f'frame_test{i}.PNG',(min_max(j)*((2**16)-1)).astype(np.uint16))


if __name__ == '__main__':
    scans = [i for i in os.listdir() if i.startswith('5') or i.startswith('16')]
    for sc in scans:
        run_scans(sc)
        print(f'------- {sc} ----- DONE -----------------')


