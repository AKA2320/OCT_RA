import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from scipy import ndimage as scp
from statsmodels.tsa.stattools import acf
from natsort import natsorted
import cv2
# import multiprocessing
from numpy.fft import fft2,fft,ifft
# from itertools import repeat
from utils import *

# os.chdir('../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/')


if __name__ == '__main__':
    # num = int(sys.argv[1])
    # path = 'timelapse_withH2O2_registered/registered/'
    path = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top/'
    print('LOADING DATA')
    data = load_data(path)
    print('DATA LOADED')

    print("SHAPE: ",data.shape)

    print('Y-MOTION')
    data = ymotion(data)
    print('Y-MOTION CORRECTED')


    masks = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    for i in tqdm(range(masks.shape[0])):
        masks[i] = slope_mask(np.squeeze(data[:,i,:,:]))

    print('MASKS GENERATED')

    os.makedirs('masks',exist_ok=True)
    for idx, img in enumerate(masks):
        cv2.imwrite(f'masks/mask_{idx}.PNG',(min_max(img)*((2**8)-1)).astype(np.uint8))

    with open(f'mask.pickle', 'wb') as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

