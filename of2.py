import os
import numpy as np
import pydicom as dicom
import cv2
from natsort import natsorted
from tqdm import tqdm
import ants

path = '/Users/rutujajangle/Library/CloudStorage/OneDrive-IndianaUniversity/pig_eyeball_time_lapse/after_apply_drop_1/pic1/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0, len(pic_paths), 10)]

pics_without_line = []
for i in tqdm(pic_paths):
    aa = dicom.dcmread(os.path.join(path, i)).pixel_array
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
    gray_images = np.array(gray_images)
    # Normalize images
    norm_images = (gray_images - gray_images.min()) / (gray_images.max() - gray_images.min())
    return norm_images

gray_images = preprocess_images(pics_without_line)

def compute_optical_flow(data):
    optical_flows = []
    prev_frame = data[0].astype(np.float32)

    for i in tqdm(range(1, len(data))):
        next_frame = data[i].astype(np.float32)
        if prev_frame.shape != next_frame.shape:
            next_frame = cv2.resize(next_frame, (prev_frame.shape[1], prev_frame.shape[0]))

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flows.append(np.mean(magnitude))
        prev_frame = next_frame

    return np.array(optical_flows)

optical_flows = compute_optical_flow(gray_images)

threshold = 1.0
change_indices = [i + 1 for i, flow in enumerate(optical_flows) if flow > threshold]

chunk_dict = {}
start_idx = 0

for idx in change_indices:
    chunk_dict[f'data_chunk_{start_idx}_{idx}'] = pics_without_line[start_idx:idx]
    start_idx = idx

chunk_dict[f'data_chunk_{start_idx}_end'] = pics_without_line[start_idx:]

moving_mask = np.zeros_like(pics_without_line[0])
moving_mask[300:450,400:600] = 1

def phase(data):  
    n = data.shape[0] // 2
    for i in tqdm(range(data.shape[0])):
        coords = cv2.phaseCorrelate(data[n].astype(np.float64), data[i].astype(np.float64))[0]
        data[i] = np.roll(data[i], int(coords[1]), axis=0)
        data[i] = np.roll(data[i], int(coords[0]), axis=1)
    return data

for key in chunk_dict:
    chunk_dict[key] = phase(chunk_dict[key])

def ants_reg(stat, mov, mask, transform_type='Affine'):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform=transform_type, mask=mov_mask)
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=reg['fwdtransforms'])
    return reg_img.numpy()

def ants_reg_mapping(stat, mov, mask, transform_type='Affine'):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    mov_mask = ants.from_numpy(mask.astype(np.float64))
    reg = ants.registration(fixed=ants2, moving=ants1, type_of_transform=transform_type, mask=mov_mask)
    return reg['fwdtransforms']

def ants_reg_translate(stat, mov, mapping, mask):
    ants1 = ants.from_numpy(mov.astype(np.float64))
    ants2 = ants.from_numpy(stat.astype(np.float64))
    reg_img = ants.apply_transforms(fixed=ants2, moving=ants1, transformlist=mapping, interpolator='nearestNeighbor')
    return reg_img.numpy()

def reg(data, mask):
    reg_after_drop1 = []
    n = data.shape[0] // 2
    for i in tqdm(range(data.shape[0])):
        regis = ants_reg(data[n], data[i], mask)
        reg_after_drop1.append(regis)
    reg_after_drop1 = np.array(reg_after_drop1)
    return reg_after_drop1

for key in chunk_dict:
    chunk_dict[key] = reg(chunk_dict[key], moving_mask)

def register_consecutive_chunks(chunks, mask):
    registered_images = []
    prev_chunk = chunks[list(chunks.keys())[0]]
    registered_images.extend(prev_chunk)

    for key in list(chunks.keys())[1:]:
        next_chunk = chunks[key]
        map_trans = ants_reg_mapping(next_chunk[0], prev_chunk[-1], mask)
        
        for img in next_chunk:
            registered_images.append(ants_reg_translate(next_chunk[0], img, map_trans, mask))
        
        prev_chunk = next_chunk
    
    return np.array(registered_images)

registered_images = register_consecutive_chunks(chunk_dict, moving_mask)

output_path = '/Users/rutujajangle/Documents/OF_affine2/'
os.makedirs(output_path, exist_ok=True)
for i, img in tqdm(enumerate(registered_images)):
    cv2.imwrite(os.path.join(output_path, f'frame_test{i}.PNG'), img.astype(np.uint16))
