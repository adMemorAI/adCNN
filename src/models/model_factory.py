# src/models/model_factory.py

from models.ResAD import ResAD
from models.CvT import CvT  # Ensure CvT is correctly implemented and imported
import inspect
import warnings
from utils.factory_utils import filter_kwargs

def get_model(model_type: str, **kwargs):
    """
    Factory function to instantiate models based on the model_type.

    Args:
        model_type (str): The type/name of the model to instantiate ('ResAD' or 'CvT').
        **kwargs: Additional keyword arguments for the model.

    Returns:
        torch.nn.Module: An instance of the specified model.
 split   """
    model_type = model_type.lower()
    if model_type == "resad":
        filtered_kwargs = filter_kwargs(ResAD, kwargs)
        return ResAD(**filtered_kwargs)
    elif model_type == "cvt":
        filtered_kwargs = filter_kwargs(CvT, kwargs)
        return CvT(**filtered_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

