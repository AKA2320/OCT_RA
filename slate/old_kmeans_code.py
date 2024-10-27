import numpy as np
import pickle
from sklearn.cluster import KMeans,MiniBatchKMeans
import pydicom as dicom
import matplotlib.pylab as plt
import os
import cv2
from natsort import natsorted
from scipy import ndimage as scp
from tqdm import tqdm
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

with open('../pig_eye_maskgen/total/final_before_mask.pickle', 'rb') as handle:
    before_mask = pickle.load(handle)[:1000]

with open('../pig_eye_maskgen/total/final_after_mask.pickle', 'rb') as handle:
    after_mask = pickle.load(handle)[:1000]

with open('../pig_eye_maskgen/total/final_after_2min_mask.pickle', 'rb') as handle:
    after2min_mask = pickle.load(handle)[:1000]


mask = np.stack((before_mask,after_mask, after2min_mask))

del before_mask
del after_mask
del after2min_mask

mask = mask[:,:,70:-70,70:-70]
mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 10

shape_mask = mask.shape
mask_flat = mask[:,:,:,:].flatten()
# del mask
# [range(0,2562146400,2)]
print('Kmeans')
kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(mask_flat.reshape(-1,1))
clust_center = sorted(kmeans.cluster_centers_)
kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center).fit(mask_flat.reshape(-1,1))
del mask_flat
print('detecting')

# with open('../pig_eye_maskgen/total/final_before_mask.pickle', 'rb') as handle:
#     before_mask = pickle.load(handle)

# with open('../pig_eye_maskgen/total/final_after_mask.pickle', 'rb') as handle:
#     after_mask = pickle.load(handle)

# with open('../pig_eye_maskgen/total/final_after_2min_mask.pickle', 'rb') as handle:
#     after2min_mask = pickle.load(handle)


# mask = np.stack((before_mask,after_mask, after2min_mask))

# del before_mask
# del after_mask
# del after2min_mask

# mask = mask[:,:,70:-70,70:-70]
# mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask))) * 10
print('predicting')
lbls = np.empty_like(mask)
lbls[:,:1000//2,:,:] = (np.array(kmeans.predict(mask[:,:1000//2,:,:].flatten().reshape(-1,1)))).reshape(shape_mask[0],1000//2,shape_mask[2],shape_mask[3])
lbls[:,1000//2:,:,:] = (np.array(kmeans.predict(mask[:,1000//2:,:,:].flatten().reshape(-1,1)))).reshape(shape_mask[0],1000//2,shape_mask[2],shape_mask[3])

del mask
lbls = (lbls-np.min(lbls))/(np.max(lbls)-np.min(lbls))

print('Loading Data')
with open(f'../pig_data/pig_60_mean_before.pickle', 'rb') as handle:
    before_60 = pickle.load(handle)[:1000]

with open(f'../pig_data/pig_60_mean_after.pickle', 'rb') as handle:
    after_60 = pickle.load(handle)[:1000]

with open(f'../pig_data/pig_60_mean_after_2min.pickle', 'rb') as handle:
    after_2min_60 = pickle.load(handle)[:1000]



before_60 = before_60[:,70:-70,70:-70]
after_60 = after_60[:,70:-70,70:-70]
after_2min_60 = after_2min_60[:,70:-70,70:-70]


all_rgb_before_60 = np.zeros((lbls[0].shape[0],before_60.shape[1],before_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls[0].shape[0])):
    all_rgb_before_60[i] = get_rgb(lbls[0][i],before_60[i]).astype(np.float32)


all_rgb_after_60 = np.zeros((lbls[1].shape[0],after_60.shape[1],after_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls[1].shape[0])):
    all_rgb_after_60[i] = get_rgb(lbls[1][i],after_60[i]).astype(np.float32)


all_rgb_after_2min_60 = np.zeros((lbls[2].shape[0],after_2min_60.shape[1],after_2min_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls[2].shape[0])):
    all_rgb_after_2min_60[i] = get_rgb(lbls[2][i],after_2min_60[i]).astype(np.float32)


for i,j in tqdm(enumerate(all_rgb_before_60)):
    cv2.imwrite(f'rgb/before/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_after_60)):
    cv2.imwrite(f'rgb/after/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_after_2min_60)):
    cv2.imwrite(f'rgb/after_2min/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)
