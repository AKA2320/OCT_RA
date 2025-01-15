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

os.chdir('/Users/akapatil/Documents/OCT/timelapse/Timelapse_without_H2O2_12_20_2024/')

if __name__ == '__main__':
    # num = int(sys.argv[1])
    # path = 'timelapse_withoutH2O2_registered/registered/'
    path = 'registered_cropped_top/'
    print('LOADING DATA')
    data = load_nested_data_pickle(path)
    print('DATA LOADED')
    print("SHAPE: ",data.shape)

    os.makedirs('avg_data_withoutH2O2_top',exist_ok=True)
    for i in range(data.shape[1]):
        cv2.imwrite(f'avg_data_withoutH2O2_top/avg_{i}.PNG',(min_max(np.mean(data[:,i,:,:],axis=0))*((2**8)-1)).astype(np.uint8))