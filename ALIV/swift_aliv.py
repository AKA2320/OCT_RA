
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



if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    tic = time()

    data_path = "../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/Timelapse_with_H2O2/registered_cropped_bottom/"
    image_list = os.listdir(data_path)#list all images address in data_path

    fourD_image_volume = load_nested_data_pickle(data_path, len(image_list))#load all image volume and combine them in one 4D np array
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
    

    with multiprocessing.Pool(processes=120, initializer=init_worker, initargs=(shared_map, shared_shape)) as pool:
        tasks = [(i, j, k, fourD_image_volume, time_index_list)
                  for i in range(fourD_shape[1])
                  for j in range(fourD_shape[2])
                  for k in range(fourD_shape[3])
        ]
        pool.map(LIV_multiprocess, tasks)
    
    
    toc = time()
    print("Time: ", toc - tic)
    array_normalized = cv2.normalize(swift_map[3, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    with open('swift.pickle', 'wb') as handle:
        pickle.dump(swift_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("swift.npy", "wb") as f:
    #     np.save(f, swift_map)