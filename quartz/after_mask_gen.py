import pydicom as dicom
import cupy as np
import os
import cv2
from natsort import natsorted
from scipy import ndimage as scp
from tqdm import tqdm
import pickle
from skimage.exposure import equalize_adapthist

with open('tmp/final_before_mask.pickle', 'rb') as handle:
    before_mask = pickle.load(handle)

with open('tmp/final_after_mask.pickle', 'rb') as handle:
    after_mask = pickle.load(handle)

with open('tmp/final_after2min_mask.pickle', 'rb') as handle:
    after2min_mask = pickle.load(handle)


mask = np.stack((before_mask,after_mask,after2min_mask))
mask = mask[:,:,70:-70,70:-70]
mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
mask = equalize_adapthist(mask,clip_limit=0.35,nbins=256)


with open(f'tmp/all_final.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)