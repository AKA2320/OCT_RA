import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import cv2 as cv
import cv2
from natsort import natsorted
from skimage.registration import phase_cross_correlation

from skimage.transform import warp
from scipy import ndimage as scp
from tqdm import tqdm

import matplotlib.pyplot as plt
from time import time

from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage.filters import gaussian

from statsmodels.tsa.stattools import acf
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D)
from dipy.align import affine_registration, register_dwi_to_template
from dipy.align.transforms import AffineTransform2D
from dipy.align.imaffine import AffineRegistration
from matplotlib.colors import hsv_to_rgb

 
# path = 'Oct_10_2023_IR_card_motion/scan 5/pic/Flat'


# path = '2D/2D_timelapse_postsolution/scan1/pic/'
# pic_paths = []

# for i in os.listdir(path):
#     if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
#         pic_paths.append(i)
# pic_paths = natsorted(pic_paths)[:1000]

# pics_without_line = []
# pics_with_line = []

# for i in tqdm(pic_paths):
#     aa = dicom.dcmread(path+i).pixel_array
#     pics_with_line.append(aa.copy())
#     point = np.argmax(np.sum(aa[:500],axis=1))
#     aa[point-30:point+50]=aa[0:80]
#     aa[800:]=5
#     pics_without_line.append(aa.copy())


rand_range = [500,0,20,910]
for k in rand_range:
    for i in tqdm(range(len(pics_without_line))):
        coords = phase_cross_correlation(pics_without_line[k],pics_without_line[i],normalization=None)[0]
        pics_without_line[i] = scp.shift(pics_without_line[i],shift = (int(coords[0]),int(coords[1])),mode='nearest',order=0)
        pics_with_line[i] = scp.shift(pics_with_line[i],shift = (int(coords[0]),int(coords[1])),mode='nearest',order = 0)


affreg = AffineRegistration(verbosity=0)
transform = AffineTransform2D()

def aff_reg(static,moving):
    affine = affreg.optimize(static, moving, transform, params0=None);
    regss = affine.transform(moving);
    return regss;

stat = pics_without_line[500]
registered_aff = []
for i in tqdm(range(0,len(pics_without_line),200),leave=False):
    registered_aff.append(aff_reg(stat,pics_without_line[i]));


for i,j in tqdm(enumerate(registered_aff)):
    cv.imwrite(path+'aligned/a_registered_aff/'+f'frame{i}.PNG',j.astype(np.uint16))