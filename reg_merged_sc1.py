from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os

import cv2
from natsort import natsorted
from skimage.registration import phase_cross_correlation

from skimage.transform import warp
from scipy import ndimage as scp
from tqdm import tqdm
from statsmodels.tsa.stattools import acf


import pickle
from scipy.fftpack import fft2, fftshift, ifft2, fft
# from matplotlib.colors import hsv_to_rgb
from skimage.color import hsv2rgb as h2r

import ants.registration as ants_register
import ants
import scipy.optimize as optz


from skimage.filters import threshold_otsu



path = '2D/2D_timelapse_postsolution/scan1/pic/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = natsorted(pic_paths)[:1000]

pics_without_line = []


for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    # pics_with_line.append(aa.copy())
    point = np.argmax(np.sum(aa[:500],axis=1))
    aa[point-30:point+50]=0
    aa[800:]=0
    pics_without_line.append(aa.copy())


data_sc1 = np.array(pics_without_line)
data_sc1 = data_sc1[range(0,data_sc1.shape[0],2)]


rand_range = [250]
for _ in rand_range:
    for i in tqdm(range(len(data_sc1)),desc='Cross-Corr'):
        coords = phase_cross_correlation(data_sc1[_],data_sc1[i],normalization=None)[0]
        # data_sc1[i] = scp.shift(data_sc1[i],shift = (int(coords[0]),int(coords[1])),mode='nearest',order=0)
        data_sc1[i] = scp.shift(data_sc1[i],shift = (coords[0],coords[1]),mode='constant',order=3)



def ants_reg(stat,mov):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    reg = ants_register(ants2,ants1,type_of_transform = 'Rigid')
    reg_img = ants.apply_transforms(ants2, ants1, reg['fwdtransforms'])
    return reg_img.numpy()

def ants_reg_transfrom(stat,mov):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    reg = ants_register(ants2,ants1,type_of_transform = 'Rigid')
    # reg_img = ants.apply_transforms(ants2, ants1, reg['fwdtransforms'])
    return reg['fwdtransforms']

def apply_reg_transfrom(stat,mov,trans):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    # reg = ants_register(ants2,ants1,type_of_transform = 'Rigid')
    reg_img = ants.apply_transforms(ants2, ants1, trans)
    return reg_img.numpy()

data_sc1 = data_sc1[:20]

manual_regs_trial = []
for i in tqdm(range(0,data_sc1.shape[0])):
    regis = ants_reg(data_sc1[5],data_sc1[i])
    manual_regs_trial.append(regis)
manual_regs_trial = np.array(manual_regs_trial)



# batches=[]
# for i in tqdm(range(0,data_sc1.shape[0],50),desc='Batch - registering'):
#     min_batch = data_sc1[i:i+50].copy()
#     for k in range(0,min_batch.shape[0]):
#         regis = ants_reg(min_batch[25],min_batch[k])
#         batches.append(regis)
# batches = np.array(batches)



# for i in tqdm(range(50,batches.shape[0],50),desc='Register - Merging'):
#     regis_trans = ants_reg_transfrom(batches[i-1],batches[i])
#     for j in range(i,i+50):
#         batches[j] = apply_reg_transfrom(batches[i-1],batches[j],regis_trans)




# with open('pickles/reg_pics_sequential.pickle', 'wb') as handle:
#     pickle.dump(registered_aff, handle, protocol=pickle.HIGHEST_PROTOCOL)
for i,j in enumerate(manual_regs_trial):
    cv2.imwrite(f'/Users/akapatil/Documents/OCT/2D/2D_timelapse_postsolution/test/frame_trail{i}.PNG',j.astype(np.uint16))