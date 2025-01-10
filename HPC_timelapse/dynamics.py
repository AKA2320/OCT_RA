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

os.chdir('../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/')


if __name__ == '__main__':
    # num = int(sys.argv[1])
    path = 'timelapse_withH2O2_registered/registered_pickles/'
    # path = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_pickles/'
    print('LOADING DATA')
    data = load_data(path)
    print('DATA LOADED')

    print("SHAPE: ",data.shape)

    print('Y-MOTION')
    data = ymotion(data)
    print('Y-MOTION CORRECTED')

    masks = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    for i in range(masks.shape[0]):
        masks[i] = slope_mask(np.squeeze(data[:,i,:,:]))

    print('MASKS GENERATED')

    os.makedirs('masks',exist_ok=True)
    for idx, img in enumerate(masks):
        cv2.imwrite(f'masks/mask_{idx}.PNG',(min_max(img)*((2**8)-1)).astype(np.uint8))

    # data_name = str(sys.argv[2])
    # batch = int(sys.argv[1])

    # tot_num_frames = num_frames(data_name)

    # if tot_num_frames-batch<120:
    #     arr = np.zeros((tot_num_frames-batch,500,1500,454),dtype=np.float32)
    # else:
    #     arr = np.zeros((120,500,1500,454),dtype=np.float32)
    
    # j=0
    # for num in range(batch,batch+120):
    #     arr[j] = load_data(data_name,num*2)
    #     if j==arr.shape[0]-1:
    #         break
    #     else:
    #         j+=1
        
    # arr = arr.astype(np.float32)

    # mask_each_batch = np.zeros((arr.shape[0],arr.shape[2],arr.shape[3]), dtype=np.float32)
    # X_shape = (arr.shape[0], arr.shape[1], arr.shape[2],arr.shape[3]) # 120x40x1500x454
    # X = multiprocessing.Array('f', X_shape[0] * X_shape[1] * X_shape[2] * X_shape[3], lock=False)

    # X_np = np.frombuffer(X, dtype=np.float32).reshape(X_shape)
    # np.copyto(X_np, arr)
    # del arr

    # with multiprocessing.Pool(processes=120) as pool:
    #     results = [pool.apply_async(image, args=(x_num, X_shape, X_np[x_num])) for x_num in range(X_shape[0])]
    #     pool.close()
    #     pool.join()

    # for r in results:
    #     m,x_idx = r.get()
    #     mask_each_batch[x_idx] = m


    with open(f'mask.pickle', 'wb') as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

