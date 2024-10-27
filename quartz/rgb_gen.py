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
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist


def get_rgb(mask,img,percent = 99.9):
    h = (mask*90).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image


print('Loading Masks')

with open('final_sc1_part1_mask.pickle', 'rb') as handle:
    scan1_part1 = pickle.load(handle)

with open('final_sc1_part2_mask.pickle', 'rb') as handle:
    scan1_part2 = pickle.load(handle)

with open('final_sc2_mask.pickle', 'rb') as handle:
    scan2 = pickle.load(handle)


mask = np.vstack((scan1_part1,scan1_part2,scan2))
mask = mask[:,20:-20,20:-20]
mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))

for mk_idx in tqdm(range(mask.shape[0])):
    mkmin = np.min(mask[mk_idx])
    mkmax = np.max(mask[mk_idx])
    mask[mk_idx] = equalize_adapthist(mask[mk_idx],clip_limit=0.4)*(mkmax-mkmin)+mkmin


scan1_part1 = mask[:385]
scan1_part2 = mask[385:760]
scan2 = mask[760:]



print('Loading Data')

with open('cow_40_temp_all_sc1_part1.pickle', 'rb') as handle:
    all_sc1_part1 = pickle.load(handle)

with open('cow_40_temp_all_sc1_part2.pickle', 'rb') as handle:
    all_sc1_part2 = pickle.load(handle)

with open('cow_40_temp_all_sc2.pickle', 'rb') as handle:
    all_sc2 = pickle.load(handle)

all_sc1_part1 = all_sc1_part1[:,20:-20,20:-20]
all_sc1_part2 = all_sc1_part2[:,20:-20,20:-20]
all_sc2 = all_sc2[:,20:-20,20:-20]

all_rgb_sc1_part1 = np.zeros((scan1_part1.shape[0],all_sc1_part1.shape[1],all_sc1_part1.shape[2],3),dtype=np.float32)
for i in tqdm(range(scan1_part1.shape[0])):
    all_rgb_sc1_part1[i] = get_rgb(scan1_part1[i],all_sc1_part1[i]).astype(np.float32)

all_rgb_sc1_part2 = np.zeros((scan1_part2.shape[0],all_sc1_part2.shape[1],all_sc1_part2.shape[2],3),dtype=np.float32)
for i in tqdm(range(scan1_part2.shape[0])):
    all_rgb_sc1_part2[i] = get_rgb(scan1_part2[i],all_sc1_part2[i]).astype(np.float32)

all_rgb_sc2 = np.zeros((scan2.shape[0],all_sc2.shape[1],all_sc2.shape[2],3),dtype=np.float32)
for i in tqdm(range(scan2.shape[0])):
    all_rgb_sc2[i] = get_rgb(scan2[i],all_sc2[i]).astype(np.float32)


for i,j in tqdm(enumerate(all_rgb_sc1_part1)):
    cv2.imwrite('rgb/sc1_part1/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_sc1_part2)):
    cv2.imwrite('rgb/sc1_part2/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_sc2)):
    cv2.imwrite('rgb/sc2/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)