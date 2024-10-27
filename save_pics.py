import pickle
import numpy as np
import cv2
import os
from tqdm import tqdm
import pydicom as dicom
from natsort import natsorted
from sklearn.decomposition import PCA

path = 'pig_eyeball/after_apply_drop_2_2min/pic1/'
pic_paths = []
for i in os.listdir(path):
    if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
        pic_paths.append(i)
pic_paths = np.array(natsorted(pic_paths))[range(0,len(pic_paths),10)]
pics_without_line = []


for i in tqdm(pic_paths):
    aa = dicom.dcmread(path+i).pixel_array
    pics_without_line.append(aa.copy())

pics_without_line = np.array(pics_without_line).astype(np.float32)
pca = PCA(n_components=500)
new_data = pca.fit_transform(pics_without_line.reshape(500,-1))

with open(f'temp_pickles_SNN/temp_pig_data_after.pickle', 'wb') as handle:
    pickle.dump(new_data.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('/Users/akapatil/Documents/OCT/after_apply_drop1_pigeyeball_data.pickle', 'rb') as handle:
#     reg_pics = pickle.load(handle)

# if not os.path.exists('/Users/akapatil/Documents/OCT/pig_eyeball/registered/after'):
#     os.mkdir('/Users/akapatil/Documents/OCT/pig_eyeball/registered/after')

# for i,j in tqdm(enumerate(reg_pics)):
#     cv2.imwrite('/Users/akapatil/Documents/OCT/pig_eyeball/registered/after/'+f'frame{i}.PNG',j.astype(np.uint16))