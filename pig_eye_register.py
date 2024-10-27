import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os

import cv2
from natsort import natsorted
from skimage.registration import phase_cross_correlation
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk
# from skimage.transform import warp
from scipy import ndimage as scp
from tqdm import tqdm


import pickle
# from scipy.fftpack import fft2, fftshift, ifft2, fft
import ants.registration as ants_register
import ants
import scipy.optimize as optz
from itertools import permutations 
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import manhattan_distances
from skimage.filters import threshold_otsu

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

with open('pickles/before_drop_pigeyeball_data.pickle', 'rb') as handle:
    regs_pics = pickle.load(handle)


moving_mask = np.zeros_like(regs_pics[250])
moving_mask[304:420,:] = 1


def ants_reg(stat,mov):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(moving_mask.astype(np.float64))
    reg = ants_register(ants2,ants1,type_of_transform = 'Translation',moving_mask = mov_mask,mask = mov_mask,mask_all_stages=True)
    reg_img = ants.apply_transforms(ants2, ants1, reg['fwdtransforms'])
    return reg_img.numpy()



reg_before_drop = []
for i in tqdm(range(0,regs_pics.shape[0])):
    regis = ants_reg(regs_pics[1200],regs_pics[i])
    reg_before_drop.append(regis)
regs_pics = np.array(reg_before_drop)

del reg_before_drop

with open('before_registered.pickle', 'wb') as handle:
    pickle.dump(reg_pics_out, handle, protocol=pickle.HIGHEST_PROTOCOL)



