import os
import numpy as np
from pydicom import dcmread
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted

path = '/Users/rutujajangle/Library/CloudStorage/OneDrive-IndianaUniversity/pig_eyeball_time_lapse/after_apply_drop_1/pic1/'

pic_paths = [i for i in os.listdir(path) if i.endswith(('.dcm', '.DCM', '.PNG'))]
pic_paths = np.array(natsorted(pic_paths))[range(0, len(pic_paths), 2)]


pics_without_line = []
for i in tqdm(pic_paths):
    img = dcmread(os.path.join(path, i)).pixel_array
    pics_without_line.append(img.copy())


pics_without_line = np.array(pics_without_line)

# Calculating SSIM between consecutive frames
ssim_values = []
for i in range(len(pics_without_line) - 1):
    ssim_value, _ = ssim(pics_without_line[i], pics_without_line[i + 1], full=True)
    ssim_values.append(ssim_value)


threshold = 0.98  
change_indices = [i + 1 for i, ssim_value in enumerate(ssim_values) if ssim_value < threshold]

# Spliting the dataset into chunks based on change indices
chunks = []
start_idx = 0
for idx in change_indices:
    chunks.append(pics_without_line[start_idx:idx])
    start_idx = idx
chunks.append(pics_without_line[start_idx:])

del pics_without_line

chunk_dict = {f'data_chunk_{i}': chunk for i, chunk in enumerate(chunks)}

from skimage.registration import phase_cross_correlation
import scipy.ndimage as scp

def phase(data):
    n = data.shape[0] // 2
    for i in tqdm(range(data.shape[0])):
        coords = phase_cross_correlation(data[n], data[i], normalization=None)[0]
        data[i] = scp.shift(data[i], shift=(coords[0], coords[1]), mode='constant', order=0)
    return data


for key in chunk_dict:
    chunk_dict[key] = phase(chunk_dict[key])
    
moving_mask = np.zeros_like(chunk_dict['data_chunk_0'][0])
moving_mask[355:403, 521:608] = 1

#  main registration function that registers individual chunks
def ants_reg(stat, mov, moving_mask):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(moving_mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform='Translation', moving_mask=mov_mask, mask=mov_mask, mask_all_stages=True)
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=reg['fwdtransforms'])
    return reg_img.numpy()

# for mapping between two consecutive chunks
def ants_reg_mapping(stat, mov, moving_mask):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(moving_mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform='Translation', moving_mask=mov_mask, mask=mov_mask, mask_all_stages=True)
    return reg['fwdtransforms']

# maps the mapping to the previous chunk
def ants_reg_translate(stat, mov, mapping, moving_mask):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=mapping, interpolator='nearestNeighbor')
    return reg_img.numpy()

# registration for a chunk
def reg(data, moving_mask):
    reg_after_drop1 = []
    n = data.shape[0] // 2
    for i in tqdm(range(data.shape[0])):
        regis = ants_reg(data[n], data[i], moving_mask)
        reg_after_drop1.append(regis)
    reg_after_drop1 = np.array(reg_after_drop1)
    return reg_after_drop1


result = list(chunk_dict['data_chunk_0'])

chunk_keys = list(chunk_dict.keys())
for i in range(1, len(chunk_keys)):
    prev_chunk = chunk_dict[chunk_keys[i-1]]
    current_chunk = chunk_dict[chunk_keys[i]]
    
    
    mapping = ants_reg_mapping(current_chunk[0], prev_chunk[-1], moving_mask)
  
    registered_prev_chunk = []
    for img in tqdm(prev_chunk):
        registered_prev_chunk.append(ants_reg_translate(current_chunk[0], img, mapping))
    
   
    result.extend(registered_prev_chunk)
    

    result.extend(current_chunk)

result = np.array(result)

import cv2
for i, j in tqdm(enumerate(result)):
    cv2.imwrite(os.path.join('/Users/rutujajangle/Documents/algo98/'+f'frame_test{i}.PNG'), j.astype(np.uint16))