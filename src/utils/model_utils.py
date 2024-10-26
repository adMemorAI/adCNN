import os
import torch
import wandb

def save_and_log_model(model, config, run, loss, artifact_name):
    """
    Save the model and log it as a W&B artifact.
    """
    model_path = os.path.join(config['model_dir'], f"{artifact_name}.pth")
    
    # Ensure the model directory exists
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'], exist_ok=True)
    
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    # Optionally, remove the local model file to conserve space
    os.remove(model_path)

    print(f"{artifact_name.capitalize()} model saved and logged as artifact: {artifact.name}")


def get_model_path(model_dir, pattern="ResAD_final_model_loss_*.pth"):
    """
    Retrieve the trained model file path based on a filename pattern.

    Args:
        model_dir (str): Directory where the model is saved.
        pattern (str): Filename pattern to match.

    Returns:
        str or None: Path to the model file or None if not found.
    """
    import glob
    matching_files = glob.glob(os.path.join(model_dir, pattern))
    return matching_files[0] if matching_files else None
