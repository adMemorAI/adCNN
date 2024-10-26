# src/initialize_model.py

import os
import torch
import wandb

from utils.config import load_config
import models.resad as resad
from utils.wandb_utils import setup_wandb_run

def initialize_model(config):
    """
    Initialize the model and log it as a W&B artifact.
    """
    with setup_wandb_run(config, job_type="initialize") as run:
        # Initialize the model
        model = resad.get_default_model(**config['model_params'])
        model = model.to(config['device'])

        # Create artifact
        model_artifact = wandb.Artifact(
            name="initialized_model",
            type="model",
            description="Initialized ResAD model",
            metadata=config['model_params']
        )

        # Save the model state
        model_path = "initialized_model.pth"
        torch.save(model.state_dict(), model_path)
        model_artifact.add_file(model_path)

        # Log artifact
        run.log_artifact(model_artifact)
        wandb.save(model_path)

        print(f"Initialized model saved and logged as artifact: {model_artifact.name}")
