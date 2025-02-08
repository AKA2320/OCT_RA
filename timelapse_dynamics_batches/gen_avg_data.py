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
from config import WithoutH2O2_bottom as PATHS

# os.chdir('/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/')

if __name__ == '__main__':
    # num = int(sys.argv[1])
    # path = 'timelapse_withoutH2O2_registered/registered/'
    path = PATHS.data_path
    print('LOADING DATA')
    data = load_nested_data_pickle(path)
    print('DATA LOADED')

    data = ymotion(data)
    print('Y-MOTION CORRECTED')

    print("SHAPE: ",data.shape)

    window_size = 30
    num_vols = data.shape[0] if data.shape[0] % 2 == 0 else data.shape[0] + 1
    avg_data_shape = ((num_vols - window_size) // 2, data.shape[1], data.shape[2], data.shape[3])
    avg_data = np.zeros(avg_data_shape, dtype=np.float32)
    for slice_number in range(avg_data.shape[1]):
        for batch_number, batch in enumerate(range(0, data.shape[0] - window_size, 2)):
            avg_data[batch_number,slice_number] = min_max(np.mean(data[batch:batch + window_size, slice_number, :, :],axis=0))*((2**8)-1)

    with open(PATHS.avg_data_save_pickle, 'wb') as handle:
        pickle.dump(avg_data.astype(np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # os.makedirs('avg_data_withoutH2O2_top',exist_ok=True)
    # for i in range(data.shape[1]):
    #     cv2.imwrite(f'avg_data_withoutH2O2_top/avg_{i}.PNG',(min_max(np.mean(data[:,i,:,:],axis=0))*((2**8)-1)).astype(np.uint8))