import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import skimage as ski
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
import cv2
from natsort import natsorted
from sklearn.mixture import GaussianMixture
from skimage.registration import phase_cross_correlation
from scipy import ndimage as scp
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft
import time
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist


import ants.registration as ants_register
import ants
from scipy.optimize import minimize as minz
from itertools import permutations 
from skimage.filters import threshold_otsu
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import mean_squared_error as mse




def phase(data,mask_range=None):
    n = data.shape[0]//2
    if mask_range:
        for i in tqdm(range(data.shape[0])):
            coords = phase_cross_correlation(data[n,mask_range[0]:mask_range[1],:],data[i,mask_range[0]:mask_range[1],:],normalization=None,upsample_factor=50)[0]
            data[i] = scp.shift(data[i],shift = (coords[0],coords[1]),mode='constant',order=3)
    else:
        for i in tqdm(range(data.shape[0])):
            coords = phase_cross_correlation(data[n],data[i],normalization=None)[0]
            data[i] = scp.shift(data[i],shift = (coords[0],coords[1]),mode='constant',order=3)
    return data


def mm(data):
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    return data

    
def mse_fun_tran(shif,x,y):
    tform = AffineTransform(translation=(shif[0],shif[1]))
    warped = warp(x, tform,order=3)
    return -nmi(y,warped)

    
def upsamp(data,n):
    data = pyramid_expand(data,upscale=n,mode='constant', cval=0,order=3)
    return data

def man_reg(data):
    for i in range(data.shape[0]):
        move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(data[i,700:750,:50],4)
                ,upsamp(data[data.shape[0]//2,700:750,:50],4)))['x'])/4
        tform2 = AffineTransform(translation=(move[0],move[1]))
        data[i] = warp(data[i], tform2,order=3)
    return data

def man_join_chunks(first,second):
    move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(second[0][700:750,:50],2)
                ,upsamp(first[-1][700:750,:50],2)))['x'])/2
    tform3 = AffineTransform(translation=(move[0],move[1]))
    for i in range(len(second)):
        second[i] = warp(second[i], tform3,order=3)
    return second

path = 'cow_data/scan1/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0,len(pic_paths),2)]
pic_paths = pic_paths[:810]

data_scan1 = []
for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    data_scan1.append(aa.copy())

data_scan1 = np.array(data_scan1)
data_scan1 = data_scan1.astype(np.float32)
for i in range(data_scan1.shape[0]):
    data_scan1[i] = data_scan1[i]/np.max(data_scan1[i])


data_scan1 = phase(data_scan1,mask_range= [880,940])

for n in tqdm(range(0,data_scan1.shape[0],10)):
    data_scan1[n:n+10] = man_reg(data_scan1[n:n+10])

for n in tqdm(range(0,data_scan1.shape[0]-10,10)):
    data_scan1[n+10:n+20] = man_join_chunks(data_scan1[:n+10],data_scan1[n+10:n+20])

data_scan1 = mm(data_scan1)

for i,j in tqdm(enumerate(data_scan1)):
    cv2.imwrite('cow_data/registered/scan1_part1/'+f'frame_test{i}.tiff',(j).astype(np.float32))


    
def mse_fun_tran(shif,x,y):
    tform = AffineTransform(translation=(shif[0],shif[1]))
    warped = warp(x, tform,order=3).astype(np.float32)
    return -nmi(y,warped)

    
def mse_fun_rot(rot,x,y):
    tform = AffineTransform(rotation=(rot[0]),translation= (rot[1],rot[2]))
    warped = warp(x, tform,order=3).astype(np.float32)
    return mse(y,warped)
    
def upsamp(data,n):
    data = pyramid_expand(data,upscale=n,mode='constant', cval=0,order=3)
    return data

def man_reg(data):
    for i in range(data.shape[0]):
        move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(data[i,760:800,100:180],4)
                ,upsamp(data[data.shape[0]//2,760:800,100:180],4)))['x'])/4
        tform2 = AffineTransform(translation=(move[0],move[1]))
        data[i] = warp(data[i], tform2,order=3).astype(np.float32)
    return data

def man_join_chunks(first,second):
    move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(second[0][760:800,100:180],2)
                ,upsamp(first[-1][760:800,100:180],2)))['x'])/2
    tform3 = AffineTransform(translation=(move[0],move[1]))
    for i in range(len(second)):
        second[i] = warp(second[i], tform3,order=3).astype(np.float32)
    return second


path = 'cow_data/scan1/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0,len(pic_paths),2)]
pic_paths = pic_paths[1710:]

data_scan2 = []
for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    data_scan2.append(aa.copy())

data_scan2 = np.array(data_scan2)
data_scan2 = data_scan2.astype(np.float32)
for i in range(data_scan2.shape[0]):
    data_scan2[i] = data_scan2[i]/np.max(data_scan2[i])

data_scan2 = phase(data_scan2,mask_range= [736,864])

for i in tqdm(range(10)):
    move = (minz(method='powell',fun = mse_fun_rot,x0 =(0,0,0),args = (data_scan2[i,750:1000,:]
            ,data_scan2[5,750:1000,:]))['x'])
    tform2 = AffineTransform(rotation=(move[0]),translation= (move[1],move[2]))
    data_scan2[i] = warp(data_scan2[i], tform2,order=3).astype(np.float32)

data_scan2 = phase(data_scan2,mask_range= [736,864])

for n in tqdm(range(0,data_scan2.shape[0],10)):
    data_scan2[n:n+10] = man_reg(data_scan2[n:n+10])

for n in tqdm(range(0,data_scan2.shape[0]-10,10)):
    data_scan2[n+10:n+20] = man_join_chunks(data_scan2[:n+10],data_scan2[n+10:n+20])

data_scan2 = mm(data_scan2)

for i,j in tqdm(enumerate(data_scan2)):
    cv2.imwrite('cow_data/registered/scan1_part2/'+f'frame_test{i}.tiff',(j).astype(np.float32))


    
def mse_fun_tran(shif,x,y):
    tform = AffineTransform(translation=(shif[0],shif[1]))
    warped = warp(x, tform,order=3).astype(np.float32)
    return -nmi(y,warped)

    
def mse_fun_rot(rot,x,y):
    tform = AffineTransform(rotation=(rot[0]))
    warped = warp(x, tform,order=3).astype(np.float32)
    return -nmi(y,warped)
    
def upsamp(data,n):
    data = pyramid_expand(data,upscale=n,mode='constant', cval=0,order=3)
    return data

def man_reg(data):
    for i in range(data.shape[0]):
        move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(data[i,695:755,:75],4)
                ,upsamp(data[data.shape[0]//2,695:755,:75],4)))['x'])/4
        tform2 = AffineTransform(translation=(move[0],move[1]))
        data[i] = warp(data[i], tform2,order=3).astype(np.float32)
    return data

def man_join_chunks(first,second):
    move = (minz(method='powell',fun = mse_fun_tran,x0 =(0,0),args = (upsamp(second[0][695:755,:75],2)
                ,upsamp(first[-1][695:755,:75],2)))['x'])/2
    tform3 = AffineTransform(translation=(move[0],move[1]))
    for i in range(len(second)):
        second[i] = warp(second[i], tform3,order=3).astype(np.float32)
    return second

path = '/cow_data/scan2/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0,len(pic_paths),2)]

data_scan3 = []
for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    data_scan3.append(aa.copy())

data_scan3 = np.array(data_scan3)
data_scan3 = data_scan3.astype(np.float32)
for i in range(data_scan3.shape[0]):
    data_scan3[i] = data_scan3[i]/np.max(data_scan3[i])

data_scan3 = phase(data_scan3,mask_range= [870,910])

for n in tqdm(range(0,data_scan3.shape[0],10)):
    data_scan3[n:n+10] = man_reg(data_scan3[n:n+10])

for n in tqdm(range(0,data_scan3.shape[0]-10,10)):
    data_scan3[n+10:n+20] = man_join_chunks(data_scan3[:n+10],data_scan3[n+10:n+20])

data_scan3 = mm(data_scan3)

for i,j in tqdm(enumerate(data_scan3)):
    cv2.imwrite('cow_data/registered/scan2/'+f'frame_test{i}.tiff',(j).astype(np.float32))