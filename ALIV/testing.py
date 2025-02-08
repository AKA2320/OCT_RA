

# import numpy as np
# import os
# import sys
# from tqdm import tqdm
# import pickle
# #from scipy import ndimage as scp
# from natsort import natsorted
# import cv2
# from time import time
# # Run the script using python testing.py 0/1/2
# # 0 means before, 1 means after, 2 means after2min data

# def load_data(path_num,range_frames=None):
#     if path_num==0:
#         path = '../../data/before/'
#     elif path_num==1:
#         path = '../../data/after/'
#     elif path_num==2:
#         path = 'D:/xiaoliu_onedrive/OneDrive - Indiana University/lab/Dynamic_OCT/registration_png/'
#     pic_paths = []
#     for i in os.listdir(path):
#         if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
#             pic_paths.append(i)
#     pic_paths = natsorted(pic_paths)
#     if range_frames:
#         pic_paths = pic_paths[range_frames-50:range_frames+50]
#     pics_without_line = []

#     for i in pic_paths:
#         aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
#         pics_without_line.append(aa.copy())

#     pics_without_line = np.array(pics_without_line).astype(np.float32)
#     zero_line_down= []
#     zero_line_up = []
#     zero_line_left = []
#     zero_line_right = []
#     for i in range(pics_without_line.shape[0]):
#         for down in range(pics_without_line[i].shape[0]-1,-1,-1):
#             if np.any(pics_without_line[i][down,:]!=0):
#                 zero_line_down.append(down)
#                 break
#         for up in range(0,pics_without_line[i].shape[0]):
#             if np.any(pics_without_line[i][up,:]!=0):
#                 zero_line_up.append(up)
#                 break
#         for left in range(0,pics_without_line[i].shape[1]):
#             if np.any(pics_without_line[i][:,left]!=0):
#                 zero_line_left.append(left)
#                 break
#         for right in range(pics_without_line[i].shape[1]-1,-1,-1):
#             if np.any(pics_without_line[i][:,right]!=0):
#                 zero_line_right.append(right)
#                 break
#     zero_line_down = np.min(zero_line_down)
#     zero_line_up = np.min(zero_line_up)
#     zero_line_left = np.min(zero_line_left)
#     zero_line_right = np.min(zero_line_right)
#     pics_without_line[:,zero_line_down:,:] =  0
#     pics_without_line[:,:zero_line_up,:] =  0
#     pics_without_line[:,:,:zero_line_left] =  0
#     pics_without_line[:,:,zero_line_right:] =  0
#     return pics_without_line


# def min_max(data1):
#     if np.all(data1 == data1[0]):
#         return data1
#     else:
#         data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
#         return data1

# def slope_mask_10batch(arr,p1):
#     mask1 = np.zeros_like(arr[0],dtype=np.float32)
#     arr = arr.astype(np.float32)
#     std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)
#     arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
#     for x in range(arr.shape[1]):
#         for y in range(arr.shape[2]):
#             data1 = arr[:,x,y].astype(np.float32).copy()
#             slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
#             mask1[x, y] = -np.abs(slope1)
#     return mask1*std_mask

# tic = time()
# # num = int(sys.argv[1])
# data = load_data(2)

# data = data.astype(np.float32)
# mask = np.zeros((1,data.shape[1],data.shape[2]),dtype=np.float32)
# # j=0
# # for i in range(50,2500-50,2):
# #     mask[j] = slope_mask_10batch(data,i)
# #     j+=1
# mask = slope_mask_10batch(data, 1)
# toc = time()
# print('Done in {:.4f} seconds'.format(toc-tic))
# # with open(f'tmp/mask_{num}.pickle', 'wb') as handle:
# #     pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np #numpy for math operation
import os
from utils import load_nested_data_pickle, AVG_LIV, LIV_fun
import matplotlib.pyplot as plt
import cv2 
from scipy.optimize import curve_fit
import multiprocessing
from time import time


def AVG_LIV(oneD_data, time_index_list):
    log_data = 10 * np.log10(oneD_data + 1e-8)#convert the intensity to log data, add small number to prevent 0 from occuring  
    time_avgLIV = np.zeros((len(time_index_list), 2))#data structure to store time interval and its avgLIV
    for i in range(len(time_index_list)):
        index_data_group = time_index_list[i][1]
        num_data_group = len(index_data_group)#number of data groups that belong to this specific time interval
        LIV_group = np.zeros(num_data_group)
        for j in range(num_data_group):
            sub_log_data = log_data[index_data_group[j]]
            avglog = np.sum(sub_log_data) / (np.max(index_data_group[j])-np.min(index_data_group[j]))#time average of LIV
            sub_log_substraction = sub_log_data - avglog
            sub_LIV = np.mean(np.square(sub_log_substraction))#LIV of this particular sub dataset
            LIV_group[j] = sub_LIV
        time_avgLIV[i, :] = [time_index_list[i][0], np.mean(LIV_group)]#the average LIV of this particular time interval
    return time_avgLIV


def LIV_fun(Tw, a, tau):
    return a * (1 - np.exp(-Tw / tau))

#function in multiprocessing

def init_worker(shared_arr, shape):
    global G_array
    G_array = shared_arr
    global G_shape
    G_shape = shape

def LIV_multiprocess(args):
    i, j, k, fourD_image_volume, time_index_list = args
    # fourD_volume = np.frombuffer(shared_volume.get_obj(), dtype=np.float32).reshape(volume_shape)
    time_lapse_string = fourD_image_volume[:, i, j, k]
    array_2d = np.frombuffer(G_array.get_obj(), dtype=np.float32).reshape(G_shape)
    Average_LIV = AVG_LIV(time_lapse_string, time_index_list)
    Average_LIV[~np.isfinite(Average_LIV)] = np.mean(Average_LIV[np.isfinite(Average_LIV)])#remove infinity and NaN
    popt = curve_fit(LIV_fun, Average_LIV[:, 0], Average_LIV[:, 1], bounds=(0, np.inf))[0]
    array_2d[i, j, k] = 1 / popt[1]
# def AVG_LIV(oneD_data, time_index_list):
#     log_data = 10 * np.log10(oneD_data + 1e-8)#convert the intensity to log data, add small number to prevent 0 from occuring  
#     time_avgLIV = np.zeros((len(time_index_list), 2))#data structure to store time interval and its avgLIV
#     for i in range(len(time_index_list)):
#         index_data_group = time_index_list[i][1]
#         num_data_group = len(index_data_group)#number of data groups that belong to this specific time interval
#         LIV_group = np.zeros(num_data_group)
#         for j in range(num_data_group):
#             sub_log_data = log_data[index_data_group[j]]
#             avglog = np.sum(sub_log_data) / (np.max(index_data_group[j])-np.min(index_data_group[j]))#time average of LIV
#             sub_log_substraction = sub_log_data - avglog
#             sub_LIV = np.mean(np.square(sub_log_substraction))#LIV of this particular sub dataset
#             LIV_group[j] = sub_LIV
#         time_avgLIV[i, :] = [time_index_list[i][0], np.mean(LIV_group)]#the average LIV of this particular time interval
#     return time_avgLIV


# def LIV_fun(Tw, a, tau):
#     return a * (1 - np.exp(-Tw / tau))

# #function in multiprocessing

# def LIV_multiprocess(shared_array, shape, i, j, k, time_lapse_string, time_index_list):
#     array_2d = np.frombuffer(shared_array.get_obj()).reshape(shape)
#     Average_LIV = AVG_LIV(time_lapse_string, time_index_list)
#     Average_LIV[~np.isfinite(Average_LIV)] = np.mean(Average_LIV[np.isfinite(Average_LIV)])#remove infinity and NaN
#     popt = curve_fit(LIV_fun, Average_LIV[:, 0], Average_LIV[:, 1], bounds=(0, np.inf))[0]
#     array_2d[i, j, k] = 1 / popt[1]

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    tic = time()

    data_path = "D:/globus slate shared data Tankam Lab/without_H2O2_registered_cropped_top/"
    image_list = os.listdir(data_path)#list all images address in data_path

    fourD_image_volume = load_nested_data_pickle(data_path, len(image_list))[:, 40:50, 30:50, 200:254]#load all image volume and combine them in one 4D np array
    fourD_shape = fourD_image_volume.shape#float type 32
    number_of_scan = fourD_shape[0]


    time_index_list = []#create a list that will store both time interval and list that contain scan index comform the time interval
    for i in range(1, number_of_scan, 1):
        time_interval = i
        index_list = []
        desirable_len = len(list(range(0, number_of_scan, time_interval)))
        for j in range(time_interval):
            candidate_index = list(range(j, number_of_scan, time_interval))
            if len(candidate_index) == desirable_len:
                index_list.append(candidate_index)#avoid the case when the length of index array is short
        time_index_list.append([time_interval, index_list])


    #create shared swifit map for storing all the swifit value
    shared_map = multiprocessing.Array("f", fourD_shape[1]*fourD_shape[2]*fourD_shape[3])
    shared_shape = (fourD_shape[1],fourD_shape[2],fourD_shape[3])
    swift_map = np.frombuffer(shared_map.get_obj(), dtype=np.float32).reshape(shared_shape)
    swift_map[:] = 0
    

    # shared_volume = multiprocessing.Array("f", fourD_shape[0]*fourD_shape[1]*fourD_shape[2]*fourD_shape[3])
    # volume_shape = (fourD_shape[0],fourD_shape[1],fourD_shape[2],fourD_shape[3])
    # shared_volume_4d = np.frombuffer(shared_volume.get_obj(), dtype=np.float32).reshape(volume_shape)
    # np.copyto(shared_volume_4d, fourD_image_volume)
    #create and start multiple process to modify the shared memory
    # processse = []
    # for i in range(fourD_shape[1]):
    #     for j in range(fourD_shape[2]):
    #         for k in range(fourD_shape[3]):
    #             p = multiprocessing.Process(target=LIV_multiprocess, args=(shared_map, shared_shape, i, j, k, shared_volume, volume_shape, time_index_list))
    #             processse.append(p)
    #             p.start()
    # for p in processse:
    #     p.join()

    with multiprocessing.Pool(processes=4, initializer=init_worker, initargs=(shared_map, shared_shape)) as pool:
        tasks = [(i, j, k, fourD_image_volume, time_index_list)
                  for i in range(fourD_shape[1])
                  for j in range(fourD_shape[2])
                  for k in range(fourD_shape[3])
        ]
        pool.map(LIV_multiprocess, tasks)
    
    
    toc = time()
    print("Time: ", toc - tic)
    array_normalized = cv2.normalize(swift_map[3, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    with open("swift.npy", "wb") as f:
        np.save(f, swift_map)