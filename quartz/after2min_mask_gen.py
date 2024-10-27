import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from natsort import natsorted
from sklearn.mixture import GaussianMixture
from skimage.registration import phase_cross_correlation
from scipy import ndimage as scp
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fftpack import fft2, fftshift, ifft2, fft
import time
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist

def slope_mask_10batch(arr,p1):
    mask1 = np.zeros_like(arr[0],dtype=np.float32)
    arr = arr[p1-5:p1+5,:,:].astype(np.float32).copy()
    for x in range(arr.shape[1]):
        for y in range(arr.shape[2]):
            data1 = arr[:,x,y].astype(np.float32).copy()
            if np.all(data1 == data1[0]):
                slope1 = 0
            else:
                data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
                slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1

with open('after2min.pickle', 'rb') as handle:
    data = pickle.load(handle)

mask = np.zeros((25,data.shape[1],data.shape[2]),dtype=np.float32)
print('Running Batch')
j=0
for i in tqdm(range(5,data.shape[0],10),desc='slope_mask'):
    mask[j] = slope_mask_10batch(data,i)
    j+=1

with open('after2min_mask.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

