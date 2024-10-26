
import os
import torch
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import wandb

from utils.config import load_config
from utils.data_loader import get_data_loaders
from utils.metrics import get_metrics
import models.resad as resad
from utils.wandb_utils import setup_wandb_run
from utils.model_utils import save_and_log_model
from losses.focal_loss import FocalLoss

def train_model(config):
    """
    Train the model using the initialized artifact and log the trained model.
    """
    with setup_wandb_run(config, job_type="train") as run:
        # Use the initialized model artifact
        model_artifact = run.use_artifact("initialized_model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model.pth")
        model_config = model_artifact.metadata

        # Update config with model parameters
        config['model_params'] = model_config

        # Initialize the model
        model = resad.get_default_model(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(config['device'])

        # Watch the model with W&B
        wandb.watch(model, log="all")

        # Get data loaders and class weights
        train_loader, val_loader, class_weights = get_data_loaders(config)

        # Define loss function
        criterion = FocalLoss(
            alpha=class_weights.tolist(),  # List of class weights
            gamma=config['train_params']['focal_gamma'],
            logits=True,
            reduce=True
        )

        # Define optimizer
        optimizer = get_optimizer(config, model)

        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train_params']['scheduler_factor'],
            patience=config['train_params']['scheduler_patience'],
        )

        # Define metrics
        metrics = get_metrics(config['device'])

        # Initialize tracking variables
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(config['train_params']['num_epochs']):
            wandb.log({"Epoch": epoch + 1})

            # Training Phase
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config['device'])

            # Validation Phase
            val_loss, val_metrics = validate(model, val_loader, criterion, metrics, config['device'])

            # Log metrics
            wandb.log({
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                **{f"Validation {k}": v for k, v in val_metrics.items()},
                "Learning Rate": optimizer.param_groups[0]['lr'],
            })

            # Step the scheduler
            scheduler.step(val_loss)

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                # Save and log the best model
                save_and_log_model(model, config, run, best_loss, "best_model")
                
                # Log best metrics
                run.summary["best_val_loss"] = best_loss
                run.summary["best_val_f1"] = val_metrics.get("F1-Score", None)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['train_params']['early_stopping_patience']:
                    wandb.log({"Early Stopping": True})
                    print("Early stopping triggered.")
                    break

        # Load the best model weights
        model.load_state_dict(best_model_wts, weights_only=True)

        # Save and log the final model
        save_and_log_model(model, config, run, best_loss, "final_model")

        print("Training complete. Final model saved and logged.")

def get_optimizer(config, model):
    """
    Get the optimizer based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        model (torch.nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_type = config['train_params']['optimizer'].lower()
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=1e-4)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=config['train_params']['learning_rate'], momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['train_params']['optimizer']}")

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training DataLoader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc='Training', leave=False, unit='batch'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, metrics, device):
    """
    Validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): Validation DataLoader.
        criterion (torch.nn.Module): Loss function.
        metrics (dict): Dictionary of metric objects.
        device (torch.device): Device to validate on.

    Returns:
        tuple: (average validation loss, dictionary of metric scores)
    """
    model.eval()
    val_running_loss = 0.0
    preds_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False, unit='batch'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)
            preds = (probabilities >= 0.5).float()

            preds_list.append(preds)
            labels_list.append(labels)

    avg_val_loss = val_running_loss / len(val_loader.dataset)
    preds_tensor = torch.cat(preds_list)
    labels_tensor = torch.cat(labels_list)

    metric_scores = {}
    for name, metric in metrics.items():
        score = metric(preds_tensor, labels_tensor) * 100
        metric_scores[name] = score.item()
        metric.reset()

    return avg_val_loss, metric_scores