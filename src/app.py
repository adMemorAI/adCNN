import gradio as gr
import numpy as np
import nibabel as nib
from PIL import Image
import os
import torch
from torchvision import transforms
import warnings

# Suppress any unnecessary warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load Configuration and Model
# ---------------------------

from utils.config import load_config
from models.ResAD import ResAD  # Ensure this path is correct based on your project structure

# Load configuration
config = load_config("config.yaml")

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

# Define transformation pipeline
transform_pipeline = config['transform']  # Already a Compose object

# ---------------------------
# 2. Utility Functions
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
    slice_min = slice_img.min()
    slice_max = slice_img.max()
    if slice_max - slice_min != 0:
        slice_img = (slice_img - slice_min) / (slice_max - slice_min)
    else:
        slice_img = np.zeros_like(slice_img)
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
    img_tensor = transform_pipeline(img).unsqueeze(0).to(config.get('device', 'cpu'))  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
        label = "Alzheimer" if probability >= 0.5 else "No Alzheimer"

    return label, f"{probability * 100:.2f}%"

# ---------------------------
# 3. Gradio Interface Functions
# ---------------------------

def process_nifti(file_path):
    """
    Processes the uploaded NIfTI file and prepares data for slice viewing.

    Parameters:
        file_path (str): Path to the uploaded NIfTI file.

    Returns:
        Tuple containing the number of slices and the volume data.
    """
    try:
        # Validate file extension
        if file_path.endswith('.nii'):
            pass  # Valid .nii file
        elif file_path.endswith('.nii.gz'):
            pass  # Valid .nii.gz file
        else:
            raise ValueError("Unsupported file extension. Please upload a '.nii' or '.nii.gz' file.")

        # Load the NIfTI file
        vol = nib.load(file_path).get_fdata()
        volume = vol.T  # Transpose if necessary based on your data orientation
        nb_frames = volume.shape[0]

        return nb_frames, volume

    except Exception as e:
        # In case of any error, return None and the error message
        return None, f"Error processing NIfTI file: {str(e)}"

def on_file_upload(file_path):
    """
    Callback function when a file is uploaded.

    Parameters:
        file_path (str): Path to the uploaded NIfTI file.

    Returns:
        Tuple containing updates for the image, slider, status, number of slices, and volume data.
    """
    if file_path is None:
        return (
            gr.update(value=None),
            gr.update(maximum=1, value=0, interactive=False),
            "No file uploaded.",
            None,
            None
        )
    nb_frames, volume_or_error = process_nifti(file_path)
    if nb_frames is None:
        # An error occurred during processing
        return (
            gr.update(value=None),
            gr.update(maximum=1, value=0, interactive=False),
            volume_or_error,  # volume_or_error contains the error message
            None,
            None
        )
    # Successfully processed the file
    initial_image = get_slice(volume_or_error, 0, axis=0)
    label, prob = classify_slice(initial_image)
    return (
        initial_image,  # Initial image
        gr.update(maximum=nb_frames - 1, value=0, interactive=True),
        f"File uploaded successfully! Prediction: {label} ({prob})",
        nb_frames,
        volume_or_error
    )

def update_slice(slice_index, nb_frames, volume):
    """
    Updates the displayed slice based on the slider index and classifies it.

    Parameters:
        slice_index (int): Current slice index from the slider.
        nb_frames (int): Total number of slices.
        volume (np.ndarray): The 3D volume data.

    Returns:
        Tuple containing the new image and the status message with prediction.
    """
    if nb_frames is None or volume is None:
        return "Please upload a valid NIfTI file.", gr.update(value=None), gr.update(value=None)
    try:
        slice_index = int(np.clip(slice_index, 0, nb_frames - 1))
        image = get_slice(volume, slice_index, axis=0)
        label, prob = classify_slice(image)
        return image, f"Prediction: {label} ({prob})", label, prob
    except Exception as e:
        return f"Error displaying slice: {str(e)}", gr.update(value=None), gr.update(value=None)

# ---------------------------
# 4. Build Gradio Interface
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Brain Slice Viewer with Alzheimer's Classification")
    gr.Markdown(
        """
        Upload a `.nii` or `.nii.gz` NIfTI file to visualize its 2D brain slices.
        Use the slider below to navigate through different slices.
        The model will classify each slice as having Alzheimer's Disease or not.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload NIfTI File",
                type="filepath",
                file_types=[".nii", ".gz"]  # Allow '.nii' and '.gz' extensions
            )
            slice_slider = gr.Slider(
                minimum=0,
                maximum=1,  # Will be updated after file upload
                step=1,
                label="🖼 Slice Index",
                value=0,
                interactive=False  # Disabled until a file is uploaded
            )
            status = gr.Textbox(
                label="ℹ️ Status",
                value="Please upload a NIfTI file.",
                interactive=False
            )
        with gr.Column(scale=2):
            brain_image = gr.Image(
                label="🖼 Brain Slice",
                type="pil",
                interactive=False
            )
            prediction_label = gr.Textbox(
                label="🩺 Prediction",
                value="N/A",
                interactive=False
            )
            prediction_prob = gr.Textbox(
                label="📊 Probability",
                value="0.00%",
                interactive=False
            )

    # Hidden state to store number of slices and volume data
    state_nb_frames = gr.State()
    state_volume = gr.State()

    # File upload handler
    file_input.upload(
        fn=on_file_upload,
        inputs=file_input,
        outputs=[brain_image, slice_slider, status, state_nb_frames, state_volume]
    )

    # Slider change handler
    slice_slider.change(
        fn=update_slice,
        inputs=[slice_slider, state_nb_frames, state_volume],
        outputs=[brain_image, status, prediction_label, prediction_prob]
    )

    # Initial load setup
    demo.load(
        lambda: (
            gr.update(value=None),
            gr.update(maximum=1, value=0, interactive=False),
            "Please upload a NIfTI file.",
            None,
            None
        ),
        outputs=[brain_image, slice_slider, status, state_nb_frames, state_volume]
    )

# Launch the Gradio app
demo.launch()

