import importlib
import torch
import re
import os
from configs.config import Config

def extract_model_name(model_path):
    """
    Extracts the model class name from the model file name.
    Assumes that the model class name is the first segment separated by '-', '_', or space.

    Args:
        model_path (str): Path to the model file.

    Returns:
        str: Extracted model class name.

    Raises:
        ValueError: If the model name cannot be inferred from the file name.
    """
    filename = os.path.basename(model_path)
    # Use regex to split on '-', '_', or space
    match = re.match(r'^([A-Za-z0-9]+)[-_ ]', filename)
    if match:
        return match.group(1)
    else:
        # If no separator is found, assume the entire filename without extension is the model name
        model_name, _ = os.path.splitext(filename)
        if model_name:
            return model_name
        else:
            raise ValueError(f"Cannot extract model name from file: {model_path}")

def get_model(model_path, device):
    """
    Extracts the model class name from the model_path and initializes the model.

    Args:
        model_path (str): Path to the trained model's .pth file.
        device (torch.device): The device to which the model will be moved.

    Returns:
        torch.nn.Module: An instance of the specified model.

    Raises:
        ImportError: If the model class cannot be found.
        ValueError: If the model file does not exist or cannot be loaded.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Extract model name from the model_path
    model_name = extract_model_name(model_path)

    try:
        # Assume that all models are located in the 'models' package
        module = importlib.import_module(f"models.{model_name.lower()}")
        model_class = getattr(module, model_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Model '{model_name}' not found. Ensure it exists in the 'models' package.") from e

    # Initialize the model with the provided parameters from config
    config = Config()
    model_specific_params = config.model_params.get(model_name, {})
    model = model_class()
    model.to(device)

    # Load the model state_dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Error loading the model state_dict: {e}") from e

    model.eval()
    return model

def get_default_model():
    """
    Returns the default from config
    """
    config = Config()
    model_name = config.model_name
    model_params = config.model_params.get(model_name, {})
    module = importlib.import_module(f"models.{model_name.lower()}")
    model_class = getattr(module, model_name)
    model = model_class()
    model.to(config.device)
    return model