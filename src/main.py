# src/main.py

import argparse
import sys
import wandb

from utils.config import load_config
from utils.wandb_utils import setup_wandb_run
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline")

    # Required argument
    parser.add_argument('--step', type=str, required=True, choices=['train', 'evaluate', 'visualize'],
                        help="Pipeline step to execute: train, evaluate")

    args = parser.parse_args()

    # Initialize W&B run
    wandb.init()

    # Load configurations from config.yaml and override with W&B config
    config = load_config()

    # (Optional) Remove or comment out the following line after verification
    # print("Final Merged Config:", config)  # For debugging

    # Execute the chosen step
    if args.step == 'train':
        train_model(config)
    elif args.step == 'evaluate':
        evaluate_model(config)
    else:
        print(f"Unknown step: {args.step}")
        sys.exit(1)

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()

