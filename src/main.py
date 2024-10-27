# src/main.py

import argparse
import os
import wandb

from utils.config import load_config
from initialize_model import initialize_model
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Modular ML Pipeline with W&B Artifacts")
    
    # Required argument
    parser.add_argument('--step', type=str, required=True, choices=['initialize', 'train', 'evaluate'],
                        help="Pipeline step to execute: initialize, train, evaluate")
    
    # Optional hyperparameters (match Sweep parameters)
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--dropout', type=float, help="Dropout probability")
    parser.add_argument('--focal_gamma', type=float, help="Focal loss gamma")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for optimizer")
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], help="Optimizer type")
    parser.add_argument('--model_type', type=str, help="Type of the model to use")
    parser.add_argument('--dataset_type', type=str, help="Type of the dataset to use")
    
    args = parser.parse_args()

    # Load configurations from config.yaml
    config = load_config()

    # Override config with command-line arguments if provided
    if args.model_type is not None:
        config['model']['type'] = args.model_type
    if args.dataset_type is not None:
        config['dataset']['type'] = args.dataset_type
    if args.batch_size is not None:
        config['train_params']['batch_size'] = args.batch_size
        config['model']['params']['batch_size'] = args.batch_size
    if args.dropout is not None:
        config['model']['params']['dropout_p'] = args.dropout
    if args.focal_gamma is not None:
        config['train_params']['focal_gamma'] = args.focal_gamma
        config['evaluate_params']['focal_gamma'] = args.focal_gamma
    if args.learning_rate is not None:
        config['train_params']['learning_rate'] = args.learning_rate
    if args.optimizer is not None:
        config['train_params']['optimizer'] = args.optimizer

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

if __name__ == "__main__":
    main()
