# metrics.py

from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

def get_metrics(device):
    """
    Initialize the evaluation metrics.

    Args:
        device (torch.device): Device to move metrics to.

    Returns:
        dict: Dictionary of metric objects.
    """
    metrics = {
        "Accuracy": Accuracy(task="binary").to(device),
        "Precision": Precision(task="binary").to(device),
        "Recall": Recall(task="binary").to(device),
        "F1-Score": F1Score(task="binary").to(device)
    }
    return metrics

def compute_class_weights(labels, device):
    """
    Compute class weights for handling class imbalance.

    Args:
        labels (np.ndarray): Array of labels.
        device (torch.device): Device to move the weights to.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights
