import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from natsort import natsorted
from sklearn.mixture import GaussianMixture
from skimage.registration import phase_cross_correlation
from scipy import ndimage as scp
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft
import time
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist

def load_data(path_num,range_frames=None,dis=False):
    if path_num==0:
        path = '/Users/akapatil/Documents/OCT/cow_eyeball_time_lapse_May_30_2024/registered/scan1_part1/'
    elif path_num==1:
        path = '/Users/akapatil/Documents/OCT/cow_eyeball_time_lapse_May_30_2024/registered/scan1_part2/'
    elif path_num==2:
        path = '/Users/akapatil/Documents/OCT/cow_eyeball_time_lapse_May_30_2024/registered/scan2/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG') or i.endswith('.tiff'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    if range_frames:
        pic_paths = pic_paths[range_frames-20:range_frames+20]
    pics_without_line = []

    for i in tqdm(pic_paths,desc='Loading data',disable=dis):
        aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
        # aa = dicom.dcmread(path+i).pixel_array
        pics_without_line.append(aa.copy())

    pics_without_line = np.array(pics_without_line).astype(np.float32)
    zero_line_down= []
    zero_line_up = []
    zero_line_left = []
    zero_line_right = []
    for i in tqdm(range(pics_without_line.shape[0]),desc='cleaning',disable=dis):
        for down in range(pics_without_line[i].shape[0]-1,-1,-1):
            if np.any(pics_without_line[i][down,:]!=0):
                zero_line_down.append(down)
                break
        for up in range(0,pics_without_line[i].shape[0]):
            if np.any(pics_without_line[i][up,:]!=0):
                zero_line_up.append(up)
                break
        for left in range(0,pics_without_line[i].shape[1]):
            if np.any(pics_without_line[i][:,left]!=0):
                zero_line_left.append(left)
                break
        for right in range(pics_without_line[i].shape[1]-1,-1,-1):
            if np.any(pics_without_line[i][:,right]!=0):
                zero_line_right.append(right)
                break
    zero_line_down = np.min(zero_line_down)
    zero_line_up = np.min(zero_line_up)
    zero_line_left = np.min(zero_line_left)
    zero_line_right = np.min(zero_line_right)
    pics_without_line[:,zero_line_down:,:] =  0
    pics_without_line[:,:zero_line_up,:] =  0
    pics_without_line[:,:,:zero_line_left] =  0
    pics_without_line[:,:,zero_line_right:] =  0

    return pics_without_line


def random_data(path_num,frame_num=None):
    if path_num==0:
        path = '/Users/akapatil/Documents/OCT/pig_eyeball/registered/before/'
    elif path_num==1:
        path = '/Users/akapatil/Documents/OCT/pig_eyeball/registered/after/'
    elif path_num==2:
        path = '/Users/akapatil/Documents/OCT/pig_eyeball/registered/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    # pics_without_line = []
    if frame_num:
        pic_rand = pic_paths[frame_num]
    else:
        pic_rand = pic_paths[np.random.randint(0,len(pic_paths))]
    aa = cv2.imread(path+pic_rand,cv2.IMREAD_UNCHANGED)
    pics_without_line = np.array(aa)

    return pics_without_line


def gen_liv(arr):
    def liv_calc(pp):
        return(10*(np.log10(pp+1)))
    log_pics = list(map(liv_calc,arr))
    arrays_np = np.array(log_pics)
    average_across_arrays = np.mean(arrays_np, axis=0)
    

    liv_mask = np.zeros_like(average_across_arrays)
    for j in tqdm(range(len(log_pics)),desc='Liv'):
        liv_mask += (log_pics[j] - average_across_arrays)**2
    liv_mask/=len(log_pics)
    return -1*liv_mask



def get_rgb(mask,img,percent = 99.9):
    h = (mask*90).astype(np.float32)
    s = np.full_like(mask,1).astype(np.float32)
    v = (((img - np.min(img)) / (np.max(img) - np.min(img)))*1).astype(np.float32)
    hsv_img = np.dstack((h,s,v))
    rgb_image = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    rgb_image[rgb_image<0] = 0
    return rgb_image

def find_longest_patch(arr):
    def calc_st_ed(arr,num):
        max_length = 0
        current_length = 0
        start = -1
        end = -1
        for i, elem in enumerate(arr):
            if elem == num:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    start = i - current_length
                    end = i - 1
                current_length = 0
        return start,end
    start1,end1 = calc_st_ed(arr,1)
    start2,end2 = calc_st_ed(arr,2)

    def assign(start,end,other_start):
        if start<100:
            return end
        elif end > len(arr)-100:
            return start
        elif end<other_start:
            return end
        else:
            return start

    r1 = assign(start1,end1,start2)
    r2 = assign(start2,end2,start1)


    if np.abs(r1-r2)<50:
        if r1==end1 and end2 < len(arr)-100:
            r2=end2
        elif r2==end2 and end1 < len(arr)-100:
            r1=end1
        elif r2==start2 and start1>100:
            r1=start1
        elif r1==start1 and start2>100:
            r2=start2
            
    return r1,r2

def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size) / window_size, mode='same')

def kmeans_patches(data):
    # kk1 = KMeans(n_clusters=3,n_init='auto',random_state=0)
    pca = PCA(0.99)
    # kk1 = KMeans(n_clusters=3,n_init='auto',random_state=0)
    gm1 = GaussianMixture(n_components=3, random_state=0,reg_covar=1e-5)
    pca.fit(data[range(0,data.shape[0],5)].reshape(500,-1))
    new = pca.transform(data.reshape(data.shape[0],-1))
    lbls = gm1.fit_predict(np.abs(fft(new)))
    # kk1.fit(np.abs(fft(new)))
    # kk1.fit(data[range(0,data.shape[0],5)].reshape(500,-1))
    smoothed_array = moving_average((lbls), 25)
    smoothed_array = np.where(smoothed_array>1.5,2,np.where(smoothed_array<0.5,0,np.where((0.5<=smoothed_array) & (smoothed_array<=1.5),1,smoothed_array)))
    # p1,p2 = np.sort(find_longest_patch(kk1.labels_))
    p1,p2 = np.sort(find_longest_patch(smoothed_array))
    return p1,p2


def slope_mask(arr,p1,p2):
    mask1 = np.zeros_like(arr[0],dtype=np.float32)
    mask2 = np.zeros_like(arr[0],dtype=np.float32)
    for x in tqdm(range(arr.shape[1]),desc='slope_mask'):
        for y in range(arr.shape[2]):
            data1 = arr[p1-5:p1+5, x, y].astype(np.float32)
            data2 = arr[p2-5:p2+5, x, y].astype(np.float32)
            if np.all(data1 == data1[0]):
                slope1 = 0
            else:
                data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
                slope1 = np.polyfit(range(len(data1)), data1, 1)[0]

            if np.all(data2 == data2[0]):
                slope2 = 0
            else:
                data2 = (data2-np.min(data2))/(np.max(data2)-np.min(data2))
                slope2 = np.polyfit(range(len(data2)), data2, 1)[0]
            mask1[x, y] = -np.abs(slope1)
            mask2[x, y] = -np.abs(slope2)
    return mask1,mask2


def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1

def gen_mean_img(path_num,range_frames,dis):
    mean_img = load_data(path_num,range_frames,dis)
    mean_img = (mean_img-np.min(mean_img))/(np.max(mean_img)-np.min(mean_img))
    mean_img = equalize_adapthist(np.mean(mean_img,axis=0),nbins=4000)
    return mean_img


def slope_mask_10batch(arr,p1):
    mask1 = np.zeros_like(arr[0],dtype=np.float32)
    arr = arr[p1-50:p1+50,:,:].astype(np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)
    arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
    for x in range(arr.shape[1]):
        for y in range(arr.shape[2]):
            data1 = arr[:,x,y].astype(np.float32).copy()
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*std_mask