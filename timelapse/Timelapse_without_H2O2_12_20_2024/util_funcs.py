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
from scipy.signal import find_peaks
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


def load_data(path_num,path_all = False):
    if path_all:
        path = path_all
    else:
        path = f'registered/{path_num}/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)

    temp_img = cv2.imread(path+pic_paths[0],cv2.IMREAD_UNCHANGED) 
    imgs_from_folder = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1]))
    # imgs_from_folder = []
    for i,j in enumerate(pic_paths):
        aa = cv2.imread(path+j,cv2.IMREAD_UNCHANGED)
        imgs_from_folder[i] = aa.copy()
    imgs_from_folder = imgs_from_folder.astype(np.float32)
    return imgs_from_folder


def ants_reg_mapping(stat,mov):
    ants1 = ants.from_numpy(stat.astype(np.float32))
    ants2 = ants.from_numpy(mov.astype(np.float32))
    reg = ants_register(ants1,ants2,type_of_transform = 'Translation',
                        aff_iterations=(1100, 1200, 1000, 1000))
    return reg['fwdtransforms']

def ncc(a,b):
    a = a / np.linalg.norm(a) if np.linalg.norm(a)!=0 else a / 10
    b = b / np.linalg.norm(b) if np.linalg.norm(b)!=0 else b / 10
    return np.correlate(a.flatten(), b.flatten())

def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1


def mse_fun_tran(shif,x,y):
    tform = AffineTransform(translation=(0,shif[0]))
    warped = warp(y, tform,order=3)
    return 1-ncc(x,warped)

def ants_all_trans(data,UP,DOWN):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='tr_all'):
        temp_img = data[i+1][UP:DOWN].copy()
        # PHASE
        coords = phase_cross_correlation(min_max(data[i][UP:DOWN][:,:50])
                                        ,min_max(temp_img[:,:50])
                                        ,normalization=None,upsample_factor=20)[0]
        if np.abs(coords[0])<=2:
            temp_img = warp(temp_img,AffineTransform(translation = (0,-coords[0])),order=3)
            tff = AffineTransform(translation = (0,-coords[0]))
            transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)

        # MANUAL
        temp_tform_manual = AffineTransform(translation=(0,0))
        temp_manual = temp_img.copy()
        for _ in range(5):
            move = minz(method='powell',fun = mse_fun_tran,x0 =(0),
                        args = (data[i][UP:DOWN][:,:50]
                                ,temp_manual[:,:50]))['x']
            temp_transform = AffineTransform(translation=(0,move[0]))
            temp_manual = warp(temp_manual, temp_transform,order=3)
            temp_tform_manual = np.dot(temp_tform_manual,temp_transform)
        temp_tform_manual = AffineTransform(matrix = temp_tform_manual)
        if np.abs(np.array(temp_tform_manual)[1,2])<=2:
            temp_img = warp(temp_img,temp_tform_manual,order=3)
            transforms_all[i+1:] = np.dot(transforms_all[i+1:],temp_tform_manual)
        # # ANTS
        # mat = scipy.io.loadmat(ants_reg_mapping(min_max(data[i][UP:DOWN][:,:50]),min_max(temp_img[:,:50]))[0])
        # # mat = scipy.io.loadmat(ants_reg_mapping(min_max(data[i][UP:DOWN]),min_max(data[i+1][UP:DOWN]))[0])
        # if np.abs(mat['AffineTransform_float_2_2'][-2:][0][0])<=2:
        #     tff = AffineTransform(translation = (0,mat['AffineTransform_float_2_2'][-2:][0][0]))
        #     # ar = np.vstack((mat['AffineTransform_float_2_2'].reshape(2,3,order='F'),[0,0,1]))
        #     # ar[0,2],ar[1,2] = ar[1,2],ar[0,2]
        #     # tff = AffineTransform(matrix = ar)
        #     transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)
    return transforms_all



def bottom_extract(data,mid):
    test = np.max(data.transpose(2,1,0),axis=0).copy()
    kk = fftshift(fft2(test[-(data[0].shape[0]-mid):-80]))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    max_list = np.max(kk,axis=1)
    thresh = threshold_otsu(max_list)
    mir_UP_x, mir_DOWN_x = np.where(max_list>=thresh)[0][0]+mid, np.where(max_list>=thresh)[0][-1]+mid
    UP_x,DOWN_x = ((2*mid - mir_UP_x)-(mir_DOWN_x - mir_UP_x)), (2*mid - mir_UP_x)
    return UP_x,DOWN_x,mir_UP_x,mir_DOWN_x

def top_extract(data,mid):
    test = np.max(data.transpose(2,1,0),axis=0).copy()
    bright_point = np.argmax(np.sum(test[:mid],axis=1))+30
    kk = fftshift(fft2(test[bright_point:mid]))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    max_list = np.max(kk,axis=1)
    thresh = threshold_otsu(max_list)
    UP_x, DOWN_x = np.where(max_list>=thresh)[0][0]+bright_point, np.where(max_list>=thresh)[0][-1]+bright_point
    mir_UP_x, mir_DOWN_x = 2*mid-(np.where(max_list>=thresh)[0][-1]+bright_point), 2*mid-(np.where(max_list>=thresh)[0][0]+bright_point)
    return UP_x,DOWN_x,mir_UP_x,mir_DOWN_x

def denoise_fft(data):
    kk = fftshift(fft2(data))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    return kk

# def find_mid(data):
#     n = data.shape[1]
#     mid = (np.argmax(np.sum(data[0][:n//2],axis=1)) + data[0].shape[0])//2
#     return mid

def denoise_signal(errs):
    kk = fft(errs)
    kk[10:] = 0
    kk = abs(ifft(kk))
    return kk

def non_zero_crop(a,b):
    mini = max(np.min(np.where(a[0]!=0)),np.min(np.where(b[0]!=0)))
    maxi = min(np.max(np.where(a[0]!=0)),np.max(np.where(b[0]!=0)))
    return mini, maxi

def denoise_signal1D_err_calc(errs):
    kk = fft(errs)
    kk[20:] = 0
    kk = abs(ifft(kk))
    return kk