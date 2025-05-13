# from pydicom import dcmread
# import matplotlib.pylab as plt
import numpy as np
import os
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
from natsort import natsorted
from tqdm import tqdm
from util_funcs import *
import h5py
from ultralytics import YOLO
from reg_util_funcs import *

MODEL = YOLO('/Users/akapatil/Documents/feature_extraction/yolo_feature_extraction/yolov12s_best.pt')
SURFACE_Y_PAD = 20
SURFACE_X_PAD = 10
CELLS_X_PAD = 5
DATA_SAVE_DIR = 'registered_endo/'
DATA_LOAD_DIR = 'batch1_endo'

def main(dirname, scan_num, pbar,disable_tqdm,save_detections):
    if not os.path.exists(dirname):
        raise FileNotFoundError(f"Directory {dirname} not found")
    if not os.path.exists(os.path.join(dirname, scan_num)):
        raise FileNotFoundError(f"Scan {scan_num} not found in {dirname}")
    original_data = load_data(dirname,scan_num)
    # MODEL PART
    pbar.set_description(desc = f'Loading Model for {sc}')
    static_flat = np.argmax(np.sum(original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = save_detections, project = 'Detected Areas',name = scan_num, verbose=False,classes=[0,1])
    surface_crop_coords = [i for i in res_surface[0].summary() if i['name']=='surface']
    cells_crop_coords = [i for i in res_surface[0].summary() if i['name']=='cells']
    surface_crop_coords = detect_areas(surface_crop_coords, pad_val = 20)
    cells_crop_coords = detect_areas(cells_crop_coords, pad_val = 20)
    original_data = crop_data(original_data,surface_crop_coords,cells_crop_coords)

    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=0)
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),SURFACE_Y_PAD)
    if surface_coords is None:
        print(f'NO SURFACE DETECTED: {scan_num}')
        return None
    
    # FLATTENING PART
    pbar.set_description(desc = f'Flattening {sc}.....')
    # print('SURFACE COORDS:',surface_coords)
    static_flat = np.argmax(np.sum(original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_flat,DOWN_flat = surface_coords[i,0], surface_coords[i,1]
        UP_flat = max(UP_flat,0)
        DOWN_flat = min(DOWN_flat, original_data.shape[2])
        original_data = flatten_data(original_data,UP_flat,DOWN_flat,top_surf,disable_tqdm)
        top_surf = False

    # Y-MOTION PART
    pbar.set_description(desc = f'Correcting {sc} Y-Motion.....')
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_y,DOWN_y = surface_coords[i,0], surface_coords[i,1]
        UP_y = max(UP_y,0)
        DOWN_y = min(DOWN_y, original_data.shape[2])
        original_data = y_motion_correcting(original_data,UP_y,DOWN_y,top_surf,disable_tqdm)
        top_surf = False

    # X-MOTION PART
    pbar.set_description(desc = f'Correcting {sc} X-Motion.....')
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 0)
    res_cells = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 1)
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),SURFACE_X_PAD)
    cells_coords = detect_areas(res_cells[0].summary(),CELLS_X_PAD)

    if (cells_coords is None)and (surface_coords is None):
        print(f'NO SURFACE OR CELLS DETECTED: {scan_num}')
        return
    
    enface_extraction_rows = []
    if surface_coords is not None:
        # print('SURFACE COORDS:',surface_coords)
        static_y_motion = np.argmax(np.sum(original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(1,2)))    
        errs = []
        for i in range(original_data.shape[0]):
            errs.append(ncc(original_data[static_y_motion,:,:],original_data[i,:,:])[0])
        errs = np.squeeze(errs)
        valid_args = np.squeeze(np.argwhere(errs>0.7))
        for i in range(surface_coords.shape[0]):
            val = np.argmax(np.sum(np.max(original_data[:,surface_coords[i,0]:surface_coords[i,1],:],axis=0),axis=1))
            enface_extraction_rows.append(val)
    else:
        valid_args = np.arange(original_data.shape[0])

    if cells_coords is not None:
        if cells_coords.shape[0]==1:
            UP_x, DOWN_x = (cells_coords[0,0]), (cells_coords[0,1])
        else:
            UP_x, DOWN_x = (cells_coords[:,0]), (cells_coords[:,1])
    else:
        UP_x, DOWN_x = None,None

    tr_all = ants_all_trans_x(original_data,UP_x,DOWN_x,valid_args,enface_extraction_rows,disable_tqdm)
    for i in tqdm(range(1,original_data.shape[0],2),desc='warping',disable=disable_tqdm):
        original_data[i]  = warp(original_data[i],AffineTransform(matrix=tr_all[i]),order=3)

    pbar.set_description(desc = 'Saving Data.....')
    if original_data.dtype != np.float64:
        original_data = original_data.astype(np.float64)
    folder_save = DATA_SAVE_DIR
    os.makedirs(folder_save,exist_ok=True)
    hdf5_filename = f'{folder_save}{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=original_data, compression='gzip',compression_opts=5)


if __name__ == "__main__":
    data_dirname = DATA_LOAD_DIR
    if os.path.exists(DATA_SAVE_DIR):
        done_scans = set([i for i in os.listdir(DATA_SAVE_DIR) if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    scans = natsorted(scans)
    print('REMAINING',scans)
    pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█")
    for sc in pbar:
        pbar.set_description(desc = f'Processing {sc}')
        main(data_dirname,sc,pbar,disable_tqdm = True,save_detections = False)