# dsets/dataset_factory.py

from utils.factory_utils import filter_kwargs
from dsets.oasis_kaggle import OASISKaggle
from dsets.adni import ADNI  # Import the new ADNI dataset
import inspect
import warnings

def get_dataset(config, split: str):
    """
    Factory function to instantiate datasets based on the dataset_type.

    Args:
        config (dict): Configuration dictionary.
        split (str): One of 'train', 'test', or 'all'.

    Returns:
        torch.utils.data.Dataset: An instance of the specified dataset.
    """
    d_type = config['datasets']['type'].lower()
    if d_type == "oasiskaggle":
        return OASISKaggle(config, split=split)
    elif d_type == "adni":
        return ADNI(config, split=split)
    else:
        raise ValueError(f"Unsupported dataset type: {d_type}")

