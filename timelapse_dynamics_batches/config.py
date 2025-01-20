from dataclasses import dataclass

@dataclass
class WithH2O2_top:
    # DATA PATHS
    data_path: str = '../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/Timelapse_with_H2O2/registered_cropped_top/'
    local_data_path: str = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_top/'

    # MASK SAVE DIRECTORY
    mask_save_path: str = 'batch_masks/batch_mask_withH2O2_top/'
    mask_save_file_pickle: str = 'batch_mask_withH2O2_top.pickle'

    # RGB SAVE DIRECTORY
    rgb_save_path: str = 'batch_rgb/batch_rgb_withH2O2_top/'
    rgb_save_file_pickle: str = 'rgb_withH2O2_top.pickle'

    # AVG DATA SAVE DIRECTORY
    avg_data_save_pickle: str = 'avg_data_withH2O2_top.pickle'

@dataclass
class WithH2O2_bottom:
    # DATA PATHS
    data_path: str = '../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/Timelapse_with_H2O2/registered_cropped_bottom/'
    local_data_path: str = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_with_H2O2_12_20_2024/registered_cropped_bottom/'

    # MASK SAVE DIRECTORY
    mask_save_path: str = 'batch_masks/batch_mask_withH2O2_bottom/'
    mask_save_file_pickle: str = 'batch_mask_withH2O2_bottom.pickle'

    # RGB SAVE DIRECTORY
    rgb_save_path: str = 'batch_rgb/batch_rgb_withH2O2_bottom/'
    rgb_save_file_pickle: str = 'rgb_withH2O2_bottom.pickle'

    # AVG DATA SAVE DIRECTORY
    avg_data_save_pickle: str = 'avg_data_withH2O2_bottom.pickle'

@dataclass 
class WithoutH2O2_top:
    # DATA PATHS
    data_path: str = '../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/Timelapse_without_H2O2/registered_cropped_top/'
    local_data_path: str = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_without_H2O2_12_20_2024/registered_cropped_top/'

    # MASK SAVE DIRECTORY
    mask_save_path: str = 'batch_masks/batch_mask_withoutH2O2_top/'
    mask_save_file_pickle: str = 'batch_mask_withoutH2O2_top.pickle'

    # RGB SAVE DIRECTORY
    rgb_save_path: str = 'batch_rgb/batch_rgb_withoutH2O2_top/'
    rgb_save_file_pickle: str = 'rgb_withoutH2O2_top.pickle'

    # AVG DATA SAVE DIRECTORY
    avg_data_save_pickle: str = 'avg_data_withoutH2O2_top.pickle'

@dataclass
class WithoutH2O2_bottom:
    # DATA PATHS
    data_path: str = '../../../../../../../../N/project/OCT_preproc/CELL_DYNAMICS/Timelapse_without_H2O2/registered_cropped_bottom/'
    local_data_path: str = '/Users/akapatil/Documents/OCT/timelapse/Timelapse_without_H2O2_12_20_2024/registered_cropped_bottom/'

    # MASK SAVE DIRECTORY
    mask_save_path: str = 'batch_masks/batch_mask_withoutH2O2_bottom/'
    mask_save_file_pickle: str = 'batch_mask_withoutH2O2_bottom.pickle'

    # RGB SAVE DIRECTORY
    rgb_save_path: str = 'batch_rgb/batch_rgb_withoutH2O2_bottom/'
    rgb_save_file_pickle: str = 'rgb_withoutH2O2_bottom.pickle'

    # AVG DATA SAVE DIRECTORY
    avg_data_save_pickle: str = 'avg_data_withoutH2O2_bottom.pickle'
