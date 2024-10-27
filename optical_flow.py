import os
import numpy as np
import pydicom as dicom
import cv2
from natsort import natsorted
from tqdm import tqdm
import ants


path = 'pig_eyeball/after_apply_drop_1/pic1/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0,len(pic_paths),2)]
pic_paths = pic_paths[:5]

pics_without_line = []
for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    pics_without_line.append(aa.copy())
pics_without_line = np.array(pics_without_line)

def preprocess_images(data):
    gray_images = []
    for img in data:
        if len(img.shape) == 3:  # if image is not grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        gray_images.append(gray_img)
    return np.array(gray_images)

gray_images = preprocess_images(pics_without_line)

# Compute Optical Flow
def compute_optical_flow(data):
    optical_flows = []
    prev_frame = data[0].astype(np.float32)

    for i in range(1, len(data)):
        next_frame = data[i].astype(np.float32)
        if prev_frame.shape != next_frame.shape:
            next_frame = cv2.resize(next_frame, (prev_frame.shape[1], prev_frame.shape[0]))

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flows.append(np.mean(magnitude))
        prev_frame = next_frame

    return np.array(optical_flows)

optical_flows = compute_optical_flow(gray_images)

# Determine threshold and segment dataset
threshold = 1.0  
change_indices = [i + 1 for i, flow in enumerate(optical_flows) if flow > threshold]

# dynamic chunk dictionary
chunk_dict = {}
start_idx = 0

for idx in change_indices:
    chunk_dict[f'data_chunk_{start_idx}_{idx}'] = pics_without_line[start_idx:idx]
    start_idx = idx

chunk_dict[f'data_chunk_{start_idx}_end'] = pics_without_line[start_idx:]

moving_mask = np.zeros_like(pics_without_line[0])
moving_mask[200:350, 650:900] = 1

# Phase registration
def phase(data):
    n = data.shape[0] // 2
    for i in range(data.shape[0]):
        coords = cv2.phaseCorrelate(data[n].astype(np.float64), data[i].astype(np.float64))[0]
        data[i] = np.roll(data[i], int(coords[1]), axis=0)
        data[i] = np.roll(data[i], int(coords[0]), axis=1)
    return data

# Applying phase registration to each chunk
for key in chunk_dict:
    chunk_dict[key] = phase(chunk_dict[key])

# Main registration function that registers the individual chunks
def ants_reg(stat, mov):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(moving_mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform='Translation', mask=mov_mask)
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=reg['fwdtransforms'])
    return reg_img.numpy()

# Registration function that calculates the mapping between two consecutive chunks
def ants_reg_mapping(stat, mov):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(moving_mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform='Translation', mask=mov_mask)
    return reg['fwdtransforms']

# Registration function that maps the mapping to the previous chunk
def ants_reg_translate(stat, mov, mapping):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=mapping, interpolator='nearestNeighbor')
    return reg_img.numpy()

# Running the registration for a chunk
def reg(data):
    reg_after_drop1 = []
    n = data.shape[0] // 2
    for i in range(data.shape[0]):
        regis = ants_reg(data[n], data[i])
        reg_after_drop1.append(regis)
    reg_after_drop1 = np.array(reg_after_drop1)
    return reg_after_drop1

# Register each chunk
for key in chunk_dict:
    chunk_dict[key] = reg(chunk_dict[key])

# Registering two consecutive chunks by calculating the mapping between the last image of the previous chunk and the first image of the next chunk
def register_consecutive_chunks(chunks):
    registered_images = []
    prev_chunk = chunks[list(chunks.keys())[0]]
    
    for key in list(chunks.keys())[1:]:
        next_chunk = chunks[key]
        map_trans = ants_reg_mapping(next_chunk[0], prev_chunk[-1])
        
        for img in prev_chunk:
            registered_images.append(ants_reg_translate(next_chunk[0], img, map_trans))
        for img in next_chunk:
            registered_images.append(img)
        
        prev_chunk = next_chunk
    
    return np.array(registered_images)

# Register all chunks consecutively
registered_images = register_consecutive_chunks(chunk_dict)

for i, j in tqdm(enumerate(registered_images)):
    cv2.imwrite(os.path.join('pig_eyeball/test/'+f'frame_test{i}.PNG'), j.astype(np.uint16))