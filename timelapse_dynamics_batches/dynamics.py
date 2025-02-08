import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from scipy import ndimage as scp
from statsmodels.tsa.stattools import acf
from natsort import natsorted
import cv2
from numpy.fft import fft2,fft,ifft

from utils import *
from multiprocessing import Pool, shared_memory
from config import WithoutH2O2_bottom as PATHS
# os.chdir('../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/')
# path = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top/'
# path = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top/'
# num = int(sys.argv[1])

def process_batch_shared(args):
    batch, slice_number, shm_name, window_size, data_shape = args

    shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)

    result = slope_mask(np.squeeze(data[batch:batch + window_size, slice_number, :, :]))
    shm.close()
    return result

if __name__ == '__main__':
    # path = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top/'
    path = PATHS.data_path
    print('LOADING DATA')
    data = load_nested_data_pickle(path)
    print('DATA LOADED')

    print("SHAPE: ",data.shape)

    print('Y-MOTION')
    data = ymotion(data)
    print('Y-MOTION CORRECTED')

    window_size = 30
    num_vols = data.shape[0] if data.shape[0] % 2 == 0 else data.shape[0] + 1
    masks_shape = ((num_vols - window_size) // 2, data.shape[1], data.shape[2], data.shape[3])
    masks = np.zeros(masks_shape, dtype=np.float32)

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(shared_data, data)

    tasks = []
    for slice_number in range(masks.shape[1]):
        for batch in range(0, data.shape[0] - window_size, 2):
            tasks.append((batch, slice_number, shm.name, window_size, data.shape))

    with Pool(processes=200) as pool:
        results = list(pool.imap(process_batch_shared, tasks))
    
    idx = 0
    for slice_number in range(masks.shape[1]):
        for batch_number, batch in enumerate(range(0, data.shape[0] - window_size, 2)):
            masks[batch_number, slice_number] = results[idx]
            idx += 1
    shm.close()
    shm.unlink()
    print('MASKS GENERATED')
    os.makedirs(PATHS.mask_save_path,exist_ok=True)
    with open(PATHS.mask_save_file_pickle, 'wb') as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # for idx_batch, batch in enumerate(masks):
    #     os.makedirs(f'{PATHS.mask_save_path}batch_mask_{idx_batch}',exist_ok=True)
    #     for idx_img, img in enumerate(batch):
    #         cv2.imwrite(f"{PATHS.mask_save_path}batch_mask_{idx_batch}/mask_{idx_img}.PNG",(min_max(img)*((2**8)-1)).astype(np.uint8))
    

