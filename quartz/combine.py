import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from natsort import natsorted
from skimage.exposure import equalize_adapthist

name = str(sys.argv[1])

def num_frames(path_num):
    if (path_num==0) or (path_num=='before'):
        path = '../pig_data/before/'
    elif (path_num==1) or (path_num=='after'):
        path = '../pig_data/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = '../pig_data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    return len(range(0,len(pic_paths)-40,2))


def load_data_pickle(path):
    with open('tmp/'+path, 'rb') as handle:
        mask = pickle.load(handle)
    return mask

paths = [] 
for i in os.listdir('tmp/'):
    if name in i:
        paths.append(i)
paths = natsorted(paths)

all_mask = np.zeros((num_frames(name),1000,954),dtype=np.float32)

for i,j in enumerate(paths):
    all_mask[i*70:(i*70)+70] = load_data_pickle(j)

#all_mask = all_mask[:,70:-70,70:-70]
#all_mask = (all_mask-np.min(all_mask))/(np.max(all_mask)-np.min(all_mask))
#all_mask = equalize_adapthist(all_mask,clip_limit=0.35,nbins=256)


with open(f'total/final_{name}_mask.pickle', 'wb') as handle:
    pickle.dump(all_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
