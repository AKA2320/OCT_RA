import pickle
import numpy as np
from natsort import natsorted
import matplotlib.pylab as plt
import cv2
from tqdm import tqdm
import os
import pydicom as dicom
from scipy import ndimage as scp
from skimage.registration import phase_cross_correlation

# path = '/Users/akapatil/Documents/OCT/pig_eyeball/after_apply_drop_2_2min/pic1/'
# pic_paths = []
# for i in os.listdir(path):
#     if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
#         pic_paths.append(i)
# pic_paths = natsorted(pic_paths)

# reg_pics = []
# # reg_pics = dicom.dcmread(path+pic_paths[0]).pixel_array
# for i in tqdm(pic_paths):
#     aa = dicom.dcmread(path+i).pixel_array
#     reg_pics.append(aa.copy())

# reg_pics = np.array(reg_pics)
# reg_pics_out = reg_pics[range(0,reg_pics.shape[0],2)]
# del reg_pics

# n = reg_pics_out.shape[0]//2
# for i in tqdm(range(reg_pics_out.shape[0])):
#     coords = phase_cross_correlation(reg_pics_out[n],reg_pics_out[i],normalization=None)[0]
#     # data_sc1[i] = scp.shift(data_sc1[i],shift = (int(coords[0]),int(coords[1])),mode='nearest',order=0)
#     reg_pics_out[i] = scp.shift(reg_pics_out[i],shift = (coords[0],coords[1]),mode='constant',order=0)


with open('after_apply_drop1_2min_pigeyeball_data.pickle', 'wb') as handle:
    pickle.dump(reg_pics_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
