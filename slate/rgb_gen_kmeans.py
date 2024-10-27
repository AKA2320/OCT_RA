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
    before_mask = pickle.load(handle)

with open('../pig_eye_maskgen/total/final_after_mask.pickle', 'rb') as handle:
    after_mask = pickle.load(handle)

with open('../pig_eye_maskgen/total/final_after_2min_mask.pickle', 'rb') as handle:
    after2min_mask = pickle.load(handle)[:1000]


mask = {}
mask['before'] = before_mask
mask['after'] = after_mask
mask['after_2min'] = after2min_mask

del before_mask
del after_mask
del after2min_mask

mask['before'] = mask['before'][:,70:-70,70:-70]
mask['before'] =  ((mask['before'] - np.min(mask['before'])) / (np.max(mask['before']) - np.min(mask['before']))) * 10

mask['after'] = mask['after'][:,70:-70,70:-70]
mask['after'] =  ((mask['after'] - np.min(mask['after'])) / (np.max(mask['after']) - np.min(mask['after']))) * 10

mask['after_2min'] = mask['after_2min'][:,70:-70,70:-70]
mask['after_2min'] =  ((mask['after_2min'] - np.min(mask['after_2min'])) / (np.max(mask['after_2min']) - np.min(mask['after_2min']))) * 10


mask_flat = {}
mask_flat['before'] = mask['before'].flatten().reshape(-1,1)
mask_flat['after'] = mask['after'].flatten().reshape(-1,1)
mask_flat['after_2min'] = mask['after_2min'].flatten().reshape(-1,1)
# del mask

print('Kmeans')
lbls={}


kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(mask_flat['before'])
clust_center = sorted(kmeans.cluster_centers_)
lbls['before'] = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat['before']).reshape(mask['before'].shape[0],mask['before'].shape[1],mask['before'].shape[2])

kmeans = MiniBatchKMeans(n_clusters=7, random_state=0, n_init="auto").fit(mask_flat['after'])
clust_center = sorted(kmeans.cluster_centers_)
lbls['after'] = MiniBatchKMeans(n_clusters=7, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat['after']).reshape(mask['after'].shape[0],mask['after'].shape[1],mask['after'].shape[2])


kmeans = MiniBatchKMeans(n_clusters=7, random_state=0, n_init="auto").fit(mask_flat['after_2min'])
clust_center = sorted(kmeans.cluster_centers_)
lbls['after_2min'] = MiniBatchKMeans(n_clusters=7, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat['after_2min']).reshape(mask['after_2min'].shape[0],mask['after_2min'].shape[1],mask['after_2min'].shape[2])

del mask
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
# print('predicting')
# lbls = np.empty_like(mask)
# lbls[:,:1220//2,:,:] = (np.array(kmeans.predict(mask[:,:1220//2,:,:].flatten().reshape(-1,1)))).reshape(shape_mask[0],shape_mask[1]//2,shape_mask[2],shape_mask[3])
# lbls[:,1220//2:,:,:] = (np.array(kmeans.predict(mask[:,1220//2:,:,:].flatten().reshape(-1,1)))).reshape(shape_mask[0],shape_mask[1]//2,shape_mask[2],shape_mask[3])

# del mask
lbls['before'] = np.array((lbls['before']-np.min(lbls['before']))/(np.max(lbls['before'])-np.min(lbls['before'])))
lbls['after'] = np.array((lbls['after']-np.min(lbls['after']))/(np.max(lbls['after'])-np.min(lbls['after'])))
lbls['after_2min'] = np.array((lbls['after_2min']-np.min(lbls['after_2min']))/(np.max(lbls['after_2min'])-np.min(lbls['after_2min'])))

print('Loading Data')
with open(f'../pig_data/pig_60_mean_before.pickle', 'rb') as handle:
    before_60 = pickle.load(handle)

with open(f'../pig_data/pig_60_mean_after.pickle', 'rb') as handle:
    after_60 = pickle.load(handle)

with open(f'../pig_data/pig_60_mean_after_2min.pickle', 'rb') as handle:
    after_2min_60 = pickle.load(handle)[:1000]



before_60 = before_60[:,70:-70,70:-70]
after_60 = after_60[:,70:-70,70:-70]
after_2min_60 = after_2min_60[:,70:-70,70:-70]

all_rgb_before_60 = np.zeros((lbls['before'].shape[0],before_60.shape[1],before_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls['before'].shape[0])):
    all_rgb_before_60[i] = get_rgb(lbls['before'][i],before_60[i]).astype(np.float32)


all_rgb_after_60 = np.zeros((lbls['after'].shape[0],after_60.shape[1],after_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls['after'].shape[0])):
    all_rgb_after_60[i] = get_rgb(lbls['after'][i],after_60[i]).astype(np.float32)


all_rgb_after_2min_60 = np.zeros((lbls['after_2min'].shape[0],after_2min_60.shape[1],after_2min_60.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls['after_2min'].shape[0])):
    all_rgb_after_2min_60[i] = get_rgb(lbls['after_2min'][i],after_2min_60[i]).astype(np.float32)


for i,j in tqdm(enumerate(all_rgb_before_60)):
    cv2.imwrite(f'rgb/before/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_after_60)):
    cv2.imwrite(f'rgb/after/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)

for i,j in tqdm(enumerate(all_rgb_after_2min_60)):
    cv2.imwrite(f'rgb/after_2min/'+f'masks{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)
