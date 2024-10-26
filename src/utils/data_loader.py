import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb

from dsets.oasis_kaggle import OASISKaggle
from utils.metrics import compute_class_weights

def get_data_loaders(config):
    """
    Load datasets and create DataLoaders.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (train_loader, val_loader, class_weights)
    """
    # Initialize datasets with internal splits
    train_dataset = OASISKaggle(split='train')
    val_dataset = OASISKaggle(split='test')

    # Compute class weights based on training labels
    train_labels = np.array(train_dataset.binary_labels)
    class_weights = compute_class_weights(train_labels, config['device'])

    # Log class weights to W&B
    wandb.log({"Class Weights": class_weights.tolist()})

    # Create weighted sampler for the training data loader
    sampler = create_weighted_sampler(class_weights, train_labels)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    return train_loader, val_loader, class_weights

def create_weighted_sampler(class_weights, labels):
    """
    Create a weighted sampler for the DataLoader.

    Args:
        class_weights (torch.Tensor): Tensor of class weights.
        labels (np.ndarray): Array of labels.

    Returns:
        WeightedRandomSampler: Sampler for the DataLoader.
    """
    sample_weights = class_weights[labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
