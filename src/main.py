# main.py

import argparse
import sys

from utils.config import load_config
from utils.wandb_utils import setup_wandb_run
from initialize_model import initialize_model
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Modular ML Pipeline with W&B Integration")

    # Required argument
    parser.add_argument('--step', type=str, required=True, choices=['initialize', 'train', 'evaluate', 'full'],
                        help="Pipeline step to execute: initialize, train, evaluate, full")

    # Optional hyperparameters (match Sweep parameters)
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--dropout', type=float, help="Dropout probability")
    parser.add_argument('--focal_gamma', type=float, help="Focal loss gamma")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for optimizer")
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], help="Optimizer type")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs")
    parser.add_argument('--scheduler_factor', type=float, help="Factor for learning rate scheduler")
    parser.add_argument('--scheduler_patience', type=int, help="Patience for learning rate scheduler")
    parser.add_argument('--early_stopping_patience', type=int, help="Patience for early stopping")

    args = parser.parse_args()

    # Load configurations from config.yaml
    config = load_config()

    # Override config with command-line arguments if provided
    if args.dropout is not None:
        config['model']['params']['dropout_p'] = args.dropout
    if args.batch_size is not None:
        config['train_params']['batch_size'] = args.batch_size
    if args.focal_gamma is not None:
        config['train_params']['focal_gamma'] = args.focal_gamma
    if args.learning_rate is not None:
        config['train_params']['learning_rate'] = args.learning_rate
    if args.optimizer is not None:
        config['train_params']['optimizer'] = args.optimizer
    if args.num_epochs is not None:
        config['train_params']['num_epochs'] = args.num_epochs
    if args.scheduler_factor is not None:
        config['train_params']['scheduler_factor'] = args.scheduler_factor
    if args.scheduler_patience is not None:
        config['train_params']['scheduler_patience'] = args.scheduler_patience
    if args.early_stopping_patience is not None:
        config['train_params']['early_stopping_patience'] = args.early_stopping_patience

    # Debugging: Print the final configuration
    print(f"Executing step: {args.step}")
    print(f"Final Configuration: {config}")

    # Execute the chosen step
    if args.step == 'initialize':
        initialize_model(config)
    elif args.step == 'train':
        train_model(config)
    elif args.step == 'evaluate':
        evaluate_model(config)
    elif args.step == 'full':
        initialize_model(config)
        train_model(config)
    else:
        print(f"Unknown step: {args.step}")
        sys.exit(1)

if __name__ == "__main__":
    main()
