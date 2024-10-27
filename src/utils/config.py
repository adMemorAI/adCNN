import inspect
import yaml
import os
import wandb
from torchvision import transforms

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
        wandb_config = dict(wandb.run.config)  # Correctly convert to a dict
        config = merge_configs(config, wandb_config)

    # Build the transform based on config
    config['transform'] = build_transform(config.get('transform', {}))

    return config

def merge_configs(default_config, sweep_config):
    """
    Merge sweep configuration into the default configuration.

    Args:
        default_config (dict): Default configuration dictionary.
        sweep_config (dict): Sweep configuration dictionary.

    Returns:
        dict: Merged configuration dictionary.
    """
    for key, value in sweep_config.items():
        if key in default_config:
            if isinstance(value, dict) and isinstance(default_config[key], dict):
                default_config[key] = merge_configs(default_config[key], value)
            else:
                default_config[key] = value
        else:
            default_config[key] = value
    return default_config

def build_transform(transform_config):
    """
    Build a torchvision.transforms.Compose object based on the transform configuration.

    Args:
        transform_config (dict): Dictionary containing transform parameters.

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
        transform_list.append(transforms.RandomRotation(degrees))

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

    return transforms.Compose(transform_list)

def filter_kwargs(target_class, kwargs):
    """
    Filters the kwargs to include only those accepted by the target_class's __init__ method.

    Args:
        target_class (type): The class to instantiate.
        kwargs (dict): The keyword arguments to filter.

    Returns:
        dict: Filtered keyword arguments.
    """
    sig = inspect.signature(target_class.__init__)
    # Exclude 'self'
    params = sig.parameters
    valid_params = set(params.keys()) - {'self'}

    filtered_kwargs = {}
    missing_params = []

    for name, param in params.items():
        if name == 'self':
            continue
        if name in kwargs:
            filtered_kwargs[name] = kwargs[name]
        elif param.default == inspect.Parameter.empty:
            missing_params.append(name)

    extra_kwargs = set(kwargs.keys()) - valid_params

    if missing_params:
        warnings.warn(
            f"Missing required arguments for {target_class.__name__}: {missing_params}. Using default values if available.",
            UserWarning
        )

    if extra_kwargs:
        warnings.warn(
            f"Received unexpected keyword arguments for {target_class.__name__}: {extra_kwargs}. These will be ignored.",
            UserWarning
        )

    return filtered_kwargs