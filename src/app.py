# src/app.py

import gradio as gr
import torch
from PIL import Image
import os
import warnings

from utils.config import load_config
from models.ResAD import ResAD  # Ensure this path is correct based on your project structure

import nibabel as nib
import numpy as np
from torchvision import transforms

# Suppress any unnecessary warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load Configuration
# ---------------------------

config = load_config("config.yaml")

# ---------------------------
# 2. Load the Pretrained Model
# ---------------------------

def load_trained_model(config):
    """
    Load the pretrained ResAD model based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model_type = config.get('model_type')
    if model_type != "ResAD":
        raise ValueError(f"Unsupported model type: {model_type}")

    model_params = config['model_params']['params']
    freeze_layers = model_params.get('freeze_layers', True)
    dropout_p = model_params.get('dropout_p', 0.5)

    model = ResAD(freeze_layers=freeze_layers, dropout_p=dropout_p)

    model_dir = config.get('model_dir', 'models')
    model_path = os.path.join(model_dir, 'trained_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.get('device', 'cpu'))))
    model.to(config.get('device', 'cpu'))
    model.eval()

    return model

model = load_trained_model(config)

# ---------------------------
# 3. Define Transformation Pipeline
# ---------------------------

transform = config['transform']  # Utilizing the transform built in config.py

# ---------------------------
# 4. Define Utility Functions
# ---------------------------

def load_nifti(nifti_path):
    """
    Load a NIfTI file and return the volume data.

    Args:
        nifti_path (str): Path to the NIfTI file.

    Returns:
        np.ndarray: 3D volume data.
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        return data
    except Exception as e:
        raise ValueError(f"Error loading NIfTI file: {e}")

def get_slice(volume, slice_idx, axis=0):
    """
    Extract a 2D slice from the 3D volume.

    Args:
        volume (np.ndarray): 3D volume data.
        slice_idx (int): Index of the slice to extract.
        axis (int): Axis along which to slice (0, 1, or 2).

    Returns:
        PIL.Image: 2D slice as an image.
    """
    if axis == 0:
        slice_img = volume[slice_idx, :, :]
    elif axis == 1:
        slice_img = volume[:, slice_idx, :]
    elif axis == 2:
        slice_img = volume[:, :, slice_idx]
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    slice_img = slice_img.astype(np.float32)
    # Normalize to [0, 255] for visualization
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
    slice_img = (slice_img * 255).astype(np.uint8)
    img = Image.fromarray(slice_img)
    return img

def classify_slice(img):
    """
    Classify a 2D slice image as Alzheimer's or not.

    Args:
        img (PIL.Image): Input image.

    Returns:
        tuple: Classification label and probability.
    """
    if img is None:
        return "No image selected.", "0.00%"

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(config.get('device', 'cpu'))  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
        label = "Alzheimer" if probability >= 0.5 else "No Alzheimer"

    return label, f"{probability * 100:.2f}%"

# ---------------------------
# 5. Build Gradio Interface
# ---------------------------

def visualize_and_classify(nifti_file, slice_idx, axis):
    """
    Load NIfTI file, extract the specified slice, and classify it.

    Args:
        nifti_file (gr.File): Uploaded NIfTI file.
        slice_idx (int): Index of the slice to visualize.
        axis (str): Axis along which to slice ('axial', 'coronal', 'sagittal').

    Returns:
        PIL.Image: The extracted slice image.
        str: Classification label.
        str: Probability score.
    """
    if nifti_file is None:
        return None, "No file uploaded.", "0.00%"

    volume = load_nifti(nifti_file.name)

    axis_mapping = {'axial': 0, 'coronal': 1, 'sagittal': 2}
    if axis not in axis_mapping:
        return None, "Invalid axis selected.", "0.00%"

    axis_num = axis_mapping[axis]
    num_slices = volume.shape[axis_num]

    slice_idx = int(slice_idx)
    slice_idx = min(max(slice_idx, 0), num_slices - 1)  # Clamp index

    img = get_slice(volume, slice_idx, axis=axis_num)
    label, prob = classify_slice(img)

    return img, label, prob

# ---------------------------
# 6. Initialize Gradio App
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Alzheimer's Disease Classification Demo")
    gr.Markdown(
        """
        Upload a NIfTI file, select the slicing axis, navigate through the brain slices using the slider, and classify each slice as having Alzheimer's Disease or not.
        """
    )
    
    with gr.Row():
        with gr.Column():
            nifti_input = gr.File(label="Upload NIfTI File (.nii or .nii.gz)")
            axis_dropdown = gr.Dropdown(
                choices=["axial", "coronal", "sagittal"],
                label="Slicing Axis",
                value="axial",
                interactive=True
            )
            slice_slider = gr.Slider(label="Slice Index", minimum=0, maximum=100, step=1, value=50, visible=False)
            classify_button = gr.Button("Classify Slice", visible=False)
        with gr.Column():
            image_output = gr.Image(label="Selected Slice")
            prediction_label = gr.Textbox(label="Prediction", interactive=False)
            prediction_prob = gr.Textbox(label="Probability", interactive=False)
    
    # Hidden state to store the number of slices
    num_slices_state = gr.State()
    
    # Function to update slider based on uploaded file and axis
    def update_slider(nifti_file, axis):
        if nifti_file is None:
            return gr.Slider.update(visible=False), gr.Button.update(visible=False), None
        volume = load_nifti(nifti_file.name)
        axis_mapping = {'axial': 0, 'coronal': 1, 'sagittal': 2}
        axis_num = axis_mapping.get(axis, 0)
        num_slices = volume.shape[axis_num]
        default_slice = num_slices // 2
        # Update slider's maximum and visibility
        slider_update = gr.Slider.update(maximum=num_slices - 1, value=default_slice, visible=True)
        # Make classify button visible
        classify_update = gr.Button.update(visible=True)
        return slider_update, classify_update, num_slices
    
    # When a file is uploaded or axis is changed, update the slider
    nifti_input.change(
        fn=update_slider,
        inputs=[nifti_input, axis_dropdown],
        outputs=[slice_slider, classify_button, num_slices_state]
    )
    
    axis_dropdown.change(
        fn=update_slider,
        inputs=[nifti_input, axis_dropdown],
        outputs=[slice_slider, classify_button, num_slices_state]
    )
    
    # Function to update the displayed image based on slider
    def update_image(nifti_file, slice_idx, axis, num_slices):
        if nifti_file is None:
            return None
        axis_mapping = {'axial': 0, 'coronal': 1, 'sagittal': 2}
        axis_num = axis_mapping.get(axis, 0)
        slice_idx = int(slice_idx)
        slice_idx = min(max(slice_idx, 0), num_slices - 1)
        img = get_slice(load_nifti(nifti_file.name), slice_idx, axis=axis_num)
        return img
    
    # When the slider value changes, update the image
    slice_slider.change(
        fn=update_image,
        inputs=[nifti_input, slice_slider, axis_dropdown, num_slices_state],
        outputs=image_output
    )
    
    # When the classify button is clicked, classify the current slice
    classify_button.click(
        fn=classify_slice,
        inputs=image_output,
        outputs=[prediction_label, prediction_prob]
    )
    
    demo.launch()

