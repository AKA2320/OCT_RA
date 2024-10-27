#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from natsort import natsorted
from skimage.registration import phase_cross_correlation
from scipy import ndimage as scp
from tqdm import tqdm
import pickle
import ants.registration as ants_register
import ants


# In[2]:


# Loading the dataset
path = 'OK1_V1_2023_2_7/scan13/'
pic_paths = []

for i in os.listdir(path):
    if i.endswith('.dcm') or i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)

# Naturally sort the file paths
pic_paths = np.array(natsorted(pic_paths))
#pic_paths = pic_paths[::2]

# Reading all the images
pics_without_line = []

for i in tqdm(pic_paths):
    if i.endswith('.dcm') or i.endswith('.DCM'):
        aa = dicom.dcmread(os.path.join(path, i)).pixel_array
    elif i.endswith('.PNG'):
        aa = np.array(Image.open(os.path.join(path, i)))
    pics_without_line.append(aa.copy())

# Convert the list of images to a numpy array
pics_without_line = np.array(pics_without_line)

# Clean up memory if necessary
# del pics_without_line

print(f"Loaded {len(pics_without_line)} images.")


# In[7]:


import numpy as np
from scipy.signal import correlate2d

def calculate_ncc(image1, image2):
    """Calculate the normalized cross-correlation (NCC) between two images."""
    mean1, mean2 = np.mean(image1), np.mean(image2)
    std1, std2 = np.std(image1), np.std(image2)
    normalized1 = (image1 - mean1) / std1
    normalized2 = (image2 - mean2) / std2
    ncc = np.mean(normalized1 * normalized2)
    return ncc

def split_data(data, num_chunks):
    """Split the data into the specified number of chunks."""
    chunk_size = len(data) // num_chunks
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    if len(data) % num_chunks != 0:
        chunks.append(data[num_chunks*chunk_size:])
    return chunks

def sliding_window_affine_registration(data):
    """Perform sliding window affine registration on the given data and extract similarity metric values."""
    metric_values = []
    reg_after_phase = []
    reg_after_phase.append(data[0])  # Start with the first image as it is

    for i in tqdm(range(1, len(data))):
        ants_stat = ants.from_numpy(reg_after_phase[-1].astype(np.float64))
        ants_mov = ants.from_numpy(data[i].astype(np.float64))
        
        reg = ants.registration(fixed=ants_stat, moving=ants_mov, type_of_transform='TRSAA', regIterations=2)
        
        # Apply the transformation
        reg_img = ants.apply_transforms(fixed=ants_stat, moving=ants_mov, transformlist=reg['fwdtransforms'])
        reg_img_np = reg_img.numpy()

        # Calculate NCC as the similarity metric
        ncc_value = calculate_ncc(ants_stat.numpy(), reg_img_np)
        metric_values.append(ncc_value)

        # Print the similarity metric value
        print(f"Registration {i}: NCC Value = {ncc_value}")

    return metric_values


def register_chunks(data_chunks):
    """Register each chunk and collect similarity metric values."""
    all_metric_values = []

    for i, chunk in enumerate(data_chunks):
        metric_values = sliding_window_affine_registration(chunk)
        all_metric_values.extend(metric_values)

        if i > 0:
            # Align the last image of the previous chunk to the first image of the current chunk
            prev_last_image = ants.from_numpy(data_chunks[i-1][-1].astype(np.float64))
            curr_first_image = ants.from_numpy(chunk[0].astype(np.float64))
            reg = ants.registration(fixed=prev_last_image, moving=curr_first_image, type_of_transform='TRSAA', regIterations=6)
            
            # Apply the transformation
            reg_img = ants.apply_transforms(fixed=prev_last_image, moving=curr_first_image, transformlist=reg['fwdtransforms'])
            reg_img_np = reg_img.numpy()

            # Calculate NCC as the similarity metric
            ncc_value = calculate_ncc(prev_last_image.numpy(), reg_img_np)
            all_metric_values.append(ncc_value)
            
            # Print the similarity metric value
            print(f"Inter-chunk Registration {i}: NCC Value = {ncc_value}")

    return all_metric_values


# Split data into chunks
num_chunks = 20  # Specify the number of chunks
data_chunks = split_data(pics_without_line, num_chunks)

# Collect similarity metric values
all_metric_values = register_chunks(data_chunks)

# Print all collected metric values
print("All NCC Values:", all_metric_values)




# In[ ]:





# In[3]:


def split_data(data, num_chunks):
    """Split the data into the specified number of chunks."""
    chunk_size = len(data) // num_chunks
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    if len(data) % num_chunks != 0:
        chunks.append(data[num_chunks*chunk_size:])
    return chunks

def sliding_window_affine_registration(data):
    """Perform sliding window affine registration on the given data."""
    reg_after_phase = []
    reg_after_phase.append(data[0])  # Start with the first image as it is

    for i in tqdm(range(1, len(data))):
        ants_stat = ants.from_numpy(reg_after_phase[-1].astype(np.float64))
        ants_mov = ants.from_numpy(data[i].astype(np.float64))
        reg = ants.registration(fixed=ants_stat, moving=ants_mov, type_of_transform='TRSAA', regIterations=6)
        reg_img = ants.apply_transforms(fixed=ants_stat, moving=ants_mov, transformlist=reg['fwdtransforms'])
        reg_after_phase.append(reg_img.numpy())

    return np.array(reg_after_phase)

def register_chunks(data_chunks):
    """Register each chunk and align the last image of each chunk to the first image of the next chunk."""
    registered_chunks = []

    for i, chunk in enumerate(data_chunks):
        registered_chunk = sliding_window_affine_registration(chunk)
        registered_chunks.append(registered_chunk)

        if i > 0:
            # Align the last image of the previous chunk to the first image of the current chunk
            prev_last_image = ants.from_numpy(registered_chunks[i-1][-1].astype(np.float64))
            curr_first_image = ants.from_numpy(registered_chunk[0].astype(np.float64))
            reg = ants.registration(fixed=prev_last_image, moving=curr_first_image, type_of_transform='TRSAA', regIterations=2)
            aligned_first_image = ants.apply_transforms(fixed=prev_last_image, moving=curr_first_image, transformlist=reg['fwdtransforms']).numpy()
            registered_chunks[i][0] = aligned_first_image

    # Combine all registered chunks
    registered_data = np.concatenate(registered_chunks, axis=0)
    return registered_data

# Split data into chunks
num_chunks = 10  # Specify the number of chunks
data_chunks = split_data(pics_without_line, num_chunks)

# Register each chunk and align consecutive chunks
registered_data = register_chunks(data_chunks)




# In[4]:


for i,j in tqdm(enumerate(registered_data)):
    cv2.imwrite('test/'+f'frame_test{i}.PNG',j.astype(np.uint16))


# In[5]:


del registered_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




