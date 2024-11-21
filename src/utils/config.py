# utils/config.py

import yaml
import os
from pathlib import Path
import wandb
from torchvision import transforms
import warnings

# Define the mapping dictionary
PARAM_MAPPING = {
    # Train Parameters
    'batch_size': 'train_params.batch_size',
    'learning_rate': 'train_params.learning_rate',
    'optimizer': 'train_params.optimizer',
    'focal_gamma': 'train_params.focal_gamma',
    'num_epochs': 'train_params.num_epochs',
    'scheduler_factor': 'train_params.scheduler_factor',
    'scheduler_patience': 'train_params.scheduler_patience',
    'early_stopping_patience': 'train_params.early_stopping_patience',
    
    # Model Parameters
    'model_type': 'model.type',
    'dropout': 'model.params.dropout_p',
    'image_size': 'model.params.image_size',
    'dim': 'model.params.dim',
    'depth': 'model.params.depth',
    'heads': 'model.params.heads',
    'scale_dim': 'model.params.scale_dim',
    'pool': 'model.params.pool',
    
    # Dataset Parameters
    'dataset_type': 'datasets.type',
    
    # (Optional) Transform Parameters
    'horizontal_flip': 'transform.horizontal_flip',
    'rotation': 'transform.rotation',
}

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file and merge with W&B config if available.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Override with W&B config if present
    if wandb.run and wandb.run.config:
        wandb_config = dict(wandb.run.config)  # Convert to a dict
        config = merge_configs(config, wandb_config)

    config['project_root'] = Path(__file__).resolve().parent.parent.parent

    tr = build_transform(config.get('transform', {}))
    config['transform'] = tr

    return config

def merge_configs(default_config, sweep_config):
    """
    Merge sweep configuration into the default configuration based on PARAM_MAPPING.

    Args:
        default_config (dict): Default configuration dictionary.
        sweep_config (dict): Sweep configuration dictionary.

    Returns:
        dict: Merged configuration dictionary.
    """
    for key, value in sweep_config.items():
        if key in PARAM_MAPPING:
            path = PARAM_MAPPING[key]
            set_nested_config(default_config, path, value)
        else:
            default_config[key] = value
    return default_config

def set_nested_config(config, path, value):
    """
    Set a value in the nested configuration dictionary based on the provided path.

    Args:
        config (dict): The configuration dictionary to modify.
        path (str): Dot-separated path indicating where to set the value.
        value: The value to set.
    """
    keys = path.split('.')
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    final_key = keys[-1]
    # Handle comma-separated strings for lists
    if isinstance(d.get(final_key, None), list) and isinstance(value, str):
        try:
            value = parse_list(value)
        except ValueError as e:
            pass
    d[final_key] = value

def parse_list(param):
    """Parses a comma-separated string into a list of integers or strings."""
    if isinstance(param, str):
        # Attempt to parse integers, fall back to strings if not possible
        try:
            return [int(x.strip()) for x in param.split(',')]
        except ValueError:
            return [x.strip() for x in param.split(',')]
    return param

def build_transform(transform_config):
    """
    Build a torchvision.transforms.Compose object based on the transform configuration.

    Args:
        transform_config (dict or torchvision.transforms.Compose): 
            Dictionary containing transform parameters or an already composed transform.

    Returns:
        torchvision.transforms.Compose: The composed transform.
    """

    transform_list = []

    if 'resize' in transform_config:
        resize = transform_config['resize']
        if isinstance(resize, list) and len(resize) == 2:
            transform_list.append(transforms.Resize((resize[0], resize[1])))
        else:
            raise ValueError("resize must be a list of two integers [height, width]")

    if 'grayscale' in transform_config:
        num_output_channels = transform_config['grayscale']
        transform_list.append(transforms.Grayscale(num_output_channels=num_output_channels))

    if 'horizontal_flip' in transform_config:
        p = transform_config['horizontal_flip']
        transform_list.append(transforms.RandomHorizontalFlip(p=p))

    if 'rotation' in transform_config:
        degrees = transform_config['rotation']
        # Convert to tuple for symmetric rotation
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        transform_list.append(transforms.RandomRotation(degrees, expand=False, fill=0))

    if 'color_jitter' in transform_config:
        cj_params = transform_config['color_jitter']
        transform_list.append(transforms.ColorJitter(
            brightness=cj_params.get('brightness', 0),
            contrast=cj_params.get('contrast', 0),
            saturation=cj_params.get('saturation', 0),
            hue=cj_params.get('hue', 0)
        ))

    transform_list.append(transforms.ToTensor())

    if 'normalize' in transform_config:
        mean = transform_config['normalize'].get('mean', [0.0])
        std = transform_config['normalize'].get('std', [1.0])
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    tr = transforms.Compose(transform_list)

    return tr

