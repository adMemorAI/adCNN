# dataset_factory.py

from utils.factory_utils import filter_kwargs
from dsets.oasis_kaggle import OASISKaggle
import inspect
import warnings

def get_dataset(dataset_type: str, **kwargs):
    """
    Factory function to instantiate datasets based on the dataset_type.

    Args:
        dataset_type (str): The type/name of the dataset to instantiate.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        torch.utils.data.Dataset: An instance of the specified dataset.
    """
    dataset_type = dataset_type.lower()
    if dataset_type == "oasiskaggle":
        filtered_kwargs = filter_kwargs(OASISKaggle, kwargs)
        return OASISKaggle(**filtered_kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
