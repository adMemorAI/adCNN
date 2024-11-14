# initialize_model.py

import os
import torch
import wandb

from models.model_factory import get_model
from utils.model_utils import save_and_log_model

def initialize_model(config):
    """
    Initialize the model and log it as a W&B artifact with a fixed name 'initialized_model'.
    """
    # Check if a W&B run is already active
    if wandb.run is None:
        # Initialize W&B run if not already active
        wandb.init(project=config.get('wandb_project', 'adCNN'), job_type="initialize", config=config)
        run = wandb.run
        should_finish = True
    else:
        # Use the existing run
        run = wandb.run
        should_finish = False

    try:
        # Initialize the model using the factory
        model = get_model(
            model_type=config['model']['type'],
            **config['model']['params']
        )
        model = model.to(config['device'])

        # Define the artifact
        artifact = wandb.Artifact('initialized_model', type='model')

        # Ensure the model directory exists
        os.makedirs(config['model_dir'], exist_ok=True)

        # Save the model to a temporary file
        model_path = os.path.join(config['model_dir'], 'initialized_model.pth')
        torch.save(model.state_dict(), model_path)

        # Add the model file to the artifact
        artifact.add_file(model_path)

        # Log the artifact
        run.log_artifact(artifact)

        # Optionally, remove the temporary model file to conserve space
        os.remove(model_path)

        print(f"Initialized model saved and logged as artifact: {artifact.name}")

    finally:
        # Finish the W&B run if it was started here
        if should_finish:
            wandb.finish()
