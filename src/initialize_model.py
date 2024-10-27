import os
import torch
import wandb

from utils.config import load_config
from models.model_factory import get_model
from utils.wandb_utils import setup_wandb_run
from utils.model_utils import save_and_log_model

def initialize_model(config):
    """
    Initialize the model and log it as a W&B artifact with a fixed name 'initialized_model'.
    """
    with setup_wandb_run(config, job_type="initialize") as run:
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
