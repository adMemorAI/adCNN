import inspect
import warnings

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