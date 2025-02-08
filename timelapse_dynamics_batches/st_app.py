import streamlit as st
import numpy as np
import pickle
import os

# Directory containing datasets (update this path as needed)
DATASET_DIR = "rgb_data_combined/only_self_inter"

# Function to load dataset
def load_data(file_path, enface=True):
    if enface:
        with open(file_path, 'rb') as f:
            data = pickle.load(f).transpose(0, 2, 1, 3, 4)[..., ::-1]
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)[..., ::-1]
    return data

# Main app
st.title("Image Visualization")
#st.write("Visualize and compare 5D data interactively by selecting two datasets, each with independent controls for batch and slice.")

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

                # Display the selected image
                st.image(
                    data1[batch_1, slice_index_1],
                    use_container_width=True,
                )
            else:
                st.warning("Dataset 1 not loaded or invalid.")

        # Second Dataset
        with col2:
            if data2 is not None:
                st.subheader("Dataset 2")
                batch_2 = st.slider("Select Batch (Temporal)", 0, num_batches_2 - 1, 0, key="batch2")
                slice_index_2 = st.slider("Select Slice (Depth)", 0, num_slices_2 - 1, 0, key="slice2")

                # Display the selected image
                st.image(
                    data2[batch_2, slice_index_2],
                    use_container_width=True,
                )
            else:
                st.warning("Dataset 2 not loaded or invalid.")
    else:
        st.error("No datasets found in the specified directory.")
else:
    st.error("Dataset directory not found. Please check the path.")
