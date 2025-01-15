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


def get_rgb(mask,img,percent = 99.9):
    h = (mask*180).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image


with open('mask_top_withoutH2O2.pickle', 'rb') as handle:
    loaded_mask = pickle.load(handle)

print('Masks Loaded')
print(loaded_mask.shape)

mask = {}
mask['withoutH2O2'] = loaded_mask

del loaded_mask

# mask['withoutH2O2'] = mask['withoutH2O2'][:,50:-50,50:-50]
mask['withoutH2O2'] = min_max(mask['withoutH2O2'])
# for i in range(mask['withoutH2O2'].shape[0]):
#     mask['withoutH2O2'][i] = min_max(mask['withoutH2O2'][i])
# mask['withoutH2O2'] =  ((mask['withoutH2O2'] - np.min(mask['withoutH2O2'])) / (np.max(mask['withoutH2O2']) - np.min(mask['withoutH2O2']))) * 10

mask_flat = {}
mask_flat['withoutH2O2'] = mask['withoutH2O2'].flatten().reshape(-1,1)

print('Kmeans')
lbls={}

kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto").fit(mask_flat['withoutH2O2'])
clust_center = sorted(kmeans.cluster_centers_)
lbls['withoutH2O2'] = MiniBatchKMeans(n_clusters=5, random_state=0, n_init="auto",init=clust_center).fit_predict(mask_flat['withoutH2O2']).reshape(mask['withoutH2O2'].shape[0],mask['withoutH2O2'].shape[1],mask['withoutH2O2'].shape[2])

del mask
del mask_flat
print('detecting')


# lbls['withoutH2O2'] = np.array((lbls['withoutH2O2']-np.min(lbls['withoutH2O2']))/(np.max(lbls['withoutH2O2'])-np.min(lbls['withoutH2O2'])))
lbls['withoutH2O2'] = min_max(np.array(lbls['withoutH2O2']))

print('Loading Data')

withoutH2O2_data = load_data_png('../timelapse/Timelapse_without_H2O2_12_20_2024/avg_data_withoutH2O2_top/')
withoutH2O2_data = min_max(withoutH2O2_data)
# withoutH2O2_data = withoutH2O2_data[:,50:-50,50:-50]


all_rgb_withoutH2O2_data = np.zeros((lbls['withoutH2O2'].shape[0],withoutH2O2_data.shape[1],withoutH2O2_data.shape[2],3),dtype=np.float32)
for i in tqdm(range(lbls['withoutH2O2'].shape[0])):
    all_rgb_withoutH2O2_data[i] = get_rgb(lbls['withoutH2O2'][i],withoutH2O2_data[i]).astype(np.float32)

output_images_frontview = []
output_images_enfaceview = []
os.makedirs('test/rgb_withoutH2O2_top',exist_ok=True)
for i,j in tqdm(enumerate(all_rgb_withoutH2O2_data)):
    # print(np.unique(j))
    cv2.imwrite(f'test/rgb_withoutH2O2_top/'+f'rgb{i}.PNG',cv2.cvtColor(j, cv2.COLOR_RGB2BGR)*255)
    output_images_frontview.append((j*255).astype(np.uint8))

print(all_rgb_withoutH2O2_data.shape)
for i in tqdm(range(all_rgb_withoutH2O2_data.transpose(1,0,2,3).shape[0])):
    output_images_enfaceview.append((all_rgb_withoutH2O2_data.transpose(1,0,2,3)[i]*255).astype(np.uint8))

### GIF CODE

from PIL import Image

pil_images_front = [Image.fromarray(img) for img in output_images_frontview]
pil_images_enface = [Image.fromarray(img) for img in output_images_enfaceview]

fps = 10
duration = int(1000 / fps)

os.makedirs('GIF/frontview/rgb_withoutH2O2_top',exist_ok=True)
os.makedirs('GIF/enfaceview/rgb_withoutH2O2_top',exist_ok=True)

output_path_frontview = "GIF/frontview/rgb_withoutH2O2_top/GIF_withoutH2O2_top.gif"
pil_images_front[0].save(
    output_path_frontview,
    save_all=True,
    append_images=pil_images_front[1:],
    optimize=False,
    duration=duration,  # Set based on FPS
    loop=0  # 0 means infinite loop
)

output_path_enfaceview = "GIF/enfaceview/rgb_withoutH2O2_top/GIF_withoutH2O2_top.gif"
pil_images_enface[0].save(
    output_path_enfaceview,
    save_all=True,
    append_images=pil_images_enface[1:],
    optimize=False,
    duration=duration,  # Set based on FPS
    loop=0  # 0 means infinite loop
)

print(f"GIF saved with {fps} FPS")

