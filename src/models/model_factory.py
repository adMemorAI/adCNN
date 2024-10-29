from models.resad import ResAD
import inspect
import warnings
from utils.factory_utils import filter_kwargs


def get_model(model_type: str, **kwargs):
    """
    Factory function to instantiate models based on the model_type.

    Args:
        model_type (str): The type/name of the model to instantiate.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        torch.nn.Module: An instance of the specified model.
    """
    model_type = model_type.lower()
    if model_type == "resad":
        filtered_kwargs = filter_kwargs(ResAD, kwargs)
        return ResAD(**filtered_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
