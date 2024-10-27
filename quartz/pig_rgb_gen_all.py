import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from natsort import natsorted
from scipy import ndimage as scp
from tqdm import tqdm
import pickle
import time
import sys
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist


def get_rgb(mask,img,percent = 99.9):
    h = (mask*180).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image

# name = str(sys.argv[1])

print('Loading Masks')

with open(f'../pig_eye_maskgen/total/final_before_mask.pickle', 'rb') as handle:
    scan1 = pickle.load(handle)

with open(f'../pig_eye_maskgen/total/final_after_mask.pickle', 'rb') as handle:
    scan2 = pickle.load(handle)


scan1[np.isnan(scan1)]=0
scan2[np.isnan(scan2)]=0

scan1 = scan1[:,70:-70,70:-70]
scan2 = scan2[:,70:-70,70:-70]
mask = np.stack((scan1,scan2))
mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))

for mk_idx in tqdm(range(mask[0].shape[0])):
    mkmin = np.min(mask[0][mk_idx])
    mkmax = np.max(mask[0][mk_idx])
    mask[0][mk_idx] = equalize_adapthist(mask[0][mk_idx],clip_limit=0.4,nbins=4000)*(mkmax-mkmin)+mkmin

for mk_idx in tqdm(range(mask[1].shape[0])):
    mkmin = np.min(mask[1][mk_idx])
    mkmax = np.max(mask[1][mk_idx])
    mask[1][mk_idx] = equalize_adapthist(mask[1][mk_idx],clip_limit=0.4,nbins=4000)*(mkmax-mkmin)+mkmin

scan1 = mask[0]
scan2 = mask[1]
print('Loading Data')
with open(f'../pig_data/pig_100_temp_all_before.pickle', 'rb') as handle:
    all_sc1 = pickle.load(handle)

with open(f'../pig_data/pig_100_temp_all_after.pickle', 'rb') as handle:
    all_sc2 = pickle.load(handle)

all_sc1 = all_sc1[:,70:-70,70:-70]
all_sc2 = all_sc2[:,70:-70,70:-70]

all_rgb_sc1 = np.zeros((scan1.shape[0],all_sc1.shape[1],all_sc1.shape[2],3),dtype=np.float32)
for i in tqdm(range(scan1.shape[0])):
    all_rgb_sc1[i] = get_rgb(scan1[i],all_sc1[i]).astype(np.float32)

all_rgb_sc2 = np.zeros((scan2.shape[0],all_sc2.shape[1],all_sc2.shape[2],3),dtype=np.float32)
for i in tqdm(range(scan2.shape[0])):
    all_rgb_sc2[i] = get_rgb(scan2[i],all_sc2[i]).astype(np.float32)


for i,j in tqdm(enumerate(all_rgb_sc1)):
    cv2.imwrite(f'rgb/before/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_sc2)):
    cv2.imwrite(f'rgb/after/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)