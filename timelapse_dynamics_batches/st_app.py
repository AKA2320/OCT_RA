import streamlit as st
import numpy as np
import pickle
import os
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.pyplot as plt

# Directory containing datasets (update this path as needed)
DATASET_DIR = "/Users/akapatil/Documents/dynamicOCT/swift/"
# DATASET_DIR = "/Users/akapatil/Documents/OCT/timelapse_dynamics_batches/rgb_data_combined/"

# Function to load dataset
def load_data(file_path, enface=True):
    if enface:
        with open(file_path, 'rb') as f:
            if "swift_rgb" in file_path:
                data = pickle.load(f).transpose(0, 2, 1, 3, 4)*255
            else:
                data = pickle.load(f).transpose(0, 2, 1, 3, 4)[...,::-1]
    else:
        with open(file_path, 'rb') as f:
            if "swift_rgb" in file_path:
                data = pickle.load(f)*255
            else:
                data = pickle.load(f)[...,::-1]
    return data

def adjust_hue_clip(rgb_img, min_clip_val = 0,max_clip_val = 0.7):
    hsv_3channel = rgb_to_hsv(rgb_img)
    max_val = hsv_3channel[:,:,0].max()

    # minclip
    hsv_3channel[:,:,0] = np.where(hsv_3channel[:,:,0]<min_clip_val, 0, hsv_3channel[:,:,0])

    # maxclip
    hsv_3channel[:,:,0] = np.where(hsv_3channel[:,:,0]>max_clip_val, max_val, hsv_3channel[:,:,0])

    new_rgb = hsv_to_rgb(hsv_3channel)
    fig, ax = plt.subplots(); ax.hist(hsv_3channel[:,:,0].flatten(), bins=30)
    return new_rgb.astype(np.uint8) , fig


# Main app
st.title("Image Visualization")


# Initialize session state for dataset caching
if "loaded_data_1" not in st.session_state:
    st.session_state.loaded_data_1 = None
    st.session_state.selected_file_1 = None

if "loaded_data_2" not in st.session_state:
    st.session_state.loaded_data_2 = None
    st.session_state.selected_file_2 = None

# List available datasets
if os.path.exists(DATASET_DIR):
    dataset_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pickle')]
    if dataset_files:
        # Dataset selection
        col1, col2 = st.columns(2)
        with col1:
            selected_file_1 = st.selectbox("Select First Dataset", dataset_files, key="dataset1")
            enface_1 = st.selectbox("View Mode for Dataset 1", ["enface", "front"], key="view1")
        with col2:
            selected_file_2 = st.selectbox("Select Second Dataset", dataset_files, key="dataset2")
            enface_2 = st.selectbox("View Mode for Dataset 2", ["enface", "front"], key="view2")

        # Load the first dataset if a new file is selected
        if selected_file_1 != st.session_state.selected_file_1 or enface_1 != st.session_state.get("enface_1", None):
            dataset_path_1 = os.path.join(DATASET_DIR, selected_file_1)
            try:
                st.session_state.loaded_data_1 = load_data(dataset_path_1, enface=enface_1 == "enface")
                st.session_state.selected_file_1 = selected_file_1
                st.session_state.enface_1 = enface_1
                st.success(f"Loaded first dataset: {selected_file_1}")
            except Exception as e:
                st.error(f"Failed to load first dataset: {e}")

        # Load the second dataset if a new file is selected
        if selected_file_2 != st.session_state.selected_file_2 or enface_2 != st.session_state.get("enface_2", None):
            dataset_path_2 = os.path.join(DATASET_DIR, selected_file_2)
            try:
                st.session_state.loaded_data_2 = load_data(dataset_path_2, enface=enface_2 == "enface")
                st.session_state.selected_file_2 = selected_file_2
                st.session_state.enface_2 = enface_2
                st.success(f"Loaded second dataset: {selected_file_2}")
            except Exception as e:
                st.error(f"Failed to load second dataset: {e}")

        # Access loaded datasets
        data1 = st.session_state.loaded_data_1
        data2 = st.session_state.loaded_data_2

        if data1 is not None:
            num_batches_1, num_slices_1 = data1.shape[0], data1.shape[1]

        if data2 is not None:
            num_batches_2, num_slices_2 = data2.shape[0], data2.shape[1]

        # Sliders and visualization for each dataset
        col1, col2 = st.columns(2)

        # First Dataset
        with col1:
            if data1 is not None:
                st.subheader("Dataset 1")
                batch_1 = st.slider("Select Batch (Temporal)", 0, num_batches_1 - 1, 0, key="batch1")
                slice_index_1 = st.slider("Select Slice (Depth)", 0, num_slices_1 - 1, 0, key="slice1")
                adjust_contrast_min_1 = st.slider("Adjust Contrast(MIN)", 0.0, 0.7, 0.0, key="contrast_min1")
                adjust_contrast_max_1 = st.slider("Adjust Contrast(MAX)", 0.0, 0.7, 0.7, key="contrast_max1")

                # print(adjust_hue_clip(data1[batch_1, slice_index_1],adjust_contrast_1).min(),adjust_hue_clip(data1[batch_1, slice_index_1],adjust_contrast_1).max())

                # print((data1[batch_1, slice_index_1]).min(),(data1[batch_1, slice_index_1]).max())
                # Display the selected image
                img_to_plot_1, histogram_1 = adjust_hue_clip(data1[batch_1, slice_index_1],adjust_contrast_min_1,adjust_contrast_max_1)
                st.image(
                    img_to_plot_1,
                    use_container_width=True,
                )
                st.pyplot(histogram_1,use_container_width=True)

            else:
                st.warning("Dataset 1 not loaded or invalid.")

        # Second Dataset
        with col2:
            if data2 is not None:
                st.subheader("Dataset 2")
                batch_2 = st.slider("Select Batch (Temporal)", 0, num_batches_2 - 1, 0, key="batch2")
                slice_index_2 = st.slider("Select Slice (Depth)", 0, num_slices_2 - 1, 0, key="slice2")
                adjust_contrast_min_2 = st.slider("Adjust Contrast(MIN)", 0.0, 0.7, 0.0, key="contrast_min2")
                adjust_contrast_max_2 = st.slider("Adjust Contrast(MAX)", 0.0, 0.7, 0.7, key="contrast_max2")

                img_to_plot_2, histogram_2 = adjust_hue_clip(data2[batch_2, slice_index_2],adjust_contrast_min_2,adjust_contrast_max_2)
                # Display the selected image
                st.image(
                    img_to_plot_2,
                    use_container_width=True,
                )
                st.pyplot(histogram_2,use_container_width=True)
            else:
                st.warning("Dataset 2 not loaded or invalid.")
    else:
        st.error("No datasets found in the specified directory.")
else:
    st.error("Dataset directory not found. Please check the path.")
