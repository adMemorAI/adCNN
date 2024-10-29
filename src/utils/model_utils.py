# model_utils.py

import os
import torch
import wandb
import logging

logger = logging.getLogger(__name__)

def save_and_log_model(model, config, run, loss, artifact_name):
    """
    Save the trained model and log it as a W&B artifact.

    Args:
        model (torch.nn.Module): Trained model.
        config (dict): Configuration dictionary.
        run (wandb.sdk.wandb_run.Run): Current W&B run.
        loss (float): Validation loss of the model.
        artifact_name (str): Name of the artifact (e.g., 'final_model').
    """
    # Ensure the model directory exists
    os.makedirs(config['model_dir'], exist_ok=True)

    # Define the model file path
    model_path = os.path.join(config['model_dir'], f'{artifact_name}.pth')

    # Save the model state_dict
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")

    # Create a W&B artifact
    artifact = wandb.Artifact(artifact_name, type='model')

    # Add the model file to the artifact
    artifact.add_file(model_path)

    # Log the artifact
    run.log_artifact(artifact)
    logger.info(f"Model artifact '{artifact_name}' logged to W&B")


def get_model_path(model_dir, pattern="ResAD_final_model_loss_*.pth"):
    """
    Retrieve the latest model path matching the given pattern.

    Args:
        model_dir (str): Directory where models are saved.
        pattern (str): Glob pattern to match model files.

    Returns:
        str or None: Path to the latest model file or None if not found.
    """
    import glob

    search_pattern = os.path.join(model_dir, pattern)
    files = glob.glob(search_pattern)
    if not files:
        return None
    # Assuming the latest file has the highest name lexicographically
    latest_file = sorted(files, key=os.path.getmtime, reverse=True)[0]
    return latest_file
