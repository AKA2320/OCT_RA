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
from utils import *
from config import WithH2O2_top as PATHS


def get_rgb(mask,img,percent = 99.9):
    h = (mask*180).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image


with open(PATHS.mask_save_file_pickle, 'rb') as handle:
    loaded_mask = pickle.load(handle)

print('Masks Loaded')
print(loaded_mask.shape)

mask = {}
mask['loaded_data'] = loaded_mask

del loaded_mask
mask['loaded_data'] = min_max(mask['loaded_data'])

mask_flat = {}
mask_flat['loaded_data'] = mask['loaded_data'].flatten().reshape(-1,1)

print('Running Kmeans')
lbls={}

kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(mask_flat['loaded_data'])
clust_center = sorted(kmeans.cluster_centers_)
lbls['loaded_data'] = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat['loaded_data']).reshape(mask['loaded_data'].shape)

del mask
del mask_flat
print('detecting')

lbls['loaded_data'] = min_max(np.array(lbls['loaded_data']))

print('Loading Data')

with open(avg_data_save_pickle, 'rb') as handle:
    data = pickle.load(handle)
data = min_max(data)

print(data.shape , lbls['loaded_data'].shape)


all_rgb_data = np.zeros((data.shape[0],data.shape[1],data.shape[2], data.shape[3],3),dtype=np.uint8)
for batch_num in tqdm(range(data.shape[0])):
    for slice_num in range(data.shape[1]):
        all_rgb_data[batch_num,slice_num] = (cv2.cvtColor(get_rgb(lbls['loaded_data'][batch_num,slice_num],data[batch_num,slice_num]).astype(np.float32), cv2.COLOR_RGB2BGR)*255).astype(np.uint8)

with open(PATHS.rgb_save_file_pickle, 'wb') as handle:
    pickle.dump(all_rgb_data.astype(np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)

for idx_batch, batch in enumerate(all_rgb_data):
    os.makedirs(f'{PATHS.rgb_save_path}rgb_batch_{idx_batch}',exist_ok=True)
    for idx_img, img in enumerate(batch):
        cv2.imwrite(f"{PATHS.rgb_save_path}rgb_batch_{idx_batch}/rgb_image_{idx_img}.PNG",img)
    




'''
output_images_frontview = []
output_images_enfaceview = []
os.makedirs('test/rgb_loaded_data_top',exist_ok=True)
for i,j in tqdm(enumerate(all_rgb_data)):
    # print(np.unique(j))
    cv2.imwrite(f'test/rgb_loaded_data_top/'+f'rgb{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)
    output_images_frontview.append((j*255).astype(np.uint8))

print(all_rgb_data.shape)
for i in tqdm(range(all_rgb_data.transpose(1,0,2,3).shape[0])):
    output_images_enfaceview.append((all_rgb_data.transpose(1,0,2,3)[i]*255).astype(np.uint8))


### GIF CODE

from PIL import Image

pil_images_front = [Image.fromarray(img) for img in output_images_frontview]
pil_images_enface = [Image.fromarray(img) for img in output_images_enfaceview]

fps = 10
duration = int(1000 / fps)

os.makedirs('GIF/frontview/rgb_loaded_data_top',exist_ok=True)
os.makedirs('GIF/enfaceview/rgb_loaded_data_top',exist_ok=True)

output_path_frontview = "GIF/frontview/rgb_loaded_data_top/GIF_loaded_data_top.gif"
pil_images_front[0].save(
    output_path_frontview,
    save_all=True,
    append_images=pil_images_front[1:],
    optimize=False,
    duration=duration,  # Set based on FPS
    loop=0  # 0 means infinite loop
)

output_path_enfaceview = "GIF/enfaceview/rgb_loaded_data_top/GIF_loaded_data_top.gif"
pil_images_enface[0].save(
    output_path_enfaceview,
    save_all=True,
    append_images=pil_images_enface[1:],
    optimize=False,
    duration=duration,  # Set based on FPS
    loop=0  # 0 means infinite loop
)

print(f"GIF saved with {fps} FPS")

'''