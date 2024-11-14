# wandb_utils.py

import wandb
import subprocess
import logging

def setup_wandb_run(config, job_type):
    """
    Initialize a W&B run with the given configuration and job type.

    Args:
        config (dict): Configuration dictionary.
        job_type (str): Type of job (e.g., 'initialize', 'train', 'evaluate').

    Returns:
        wandb.run: The initialized W&B run.
    """
    run = wandb.init(project=config.get('wandb_project', 'adCNN-src'), job_type=job_type, config=config)
    run.config.update({
        "git_commit": get_git_commit(),
        "device": config['device'],  # Removed '.type' attribute
    })
    return run

def get_git_commit():
    """
    Retrieve the current git commit hash.

    Returns:
        str: Git commit hash or 'unknown' if retrieval fails.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"
