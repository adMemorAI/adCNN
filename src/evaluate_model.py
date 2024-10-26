import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import wandb

from utils.config import load_config
from dsets.oasis_kaggle import OASISKaggle
import models.resad as resad
from utils.wandb_utils import setup_wandb_run
from utils.model_utils import get_model_path
from utils.metrics import get_metrics
from losses.focal_loss import FocalLoss

def evaluate_model(config):
    """
    Evaluate the trained model and log the results.
    """
    with setup_wandb_run(config, job_type="evaluate") as run:
        # Use the trained model artifact
        model_artifact = run.use_artifact("final_model:latest")
        model_dir = model_artifact.download()
        model_path = get_model_path(model_dir, pattern="ResAD_final_model_loss_*.pth")

        if not model_path:
            raise FileNotFoundError("Trained model file not found in the artifact.")

        model_config = model_artifact.metadata

        # Initialize the model
        model = resad.get_default_model(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(config['device'])
        model.eval()

        # Load test dataset
        test_dataset = OASISKaggle(split='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        # Evaluate the model
        loss, accuracy, highest_losses, hardest_examples, true_labels, predictions = evaluate(
            model, test_loader, config
        )

        # Log summary metrics
        run.summary.update({"Loss": loss, "Accuracy": accuracy})

        # Create a W&B Table for hardest examples
        table = wandb.Table(columns=["Image", "Prediction", "True Label", "Loss"])

        for img, pred, true_label, loss_val in zip(hardest_examples, predictions, true_labels, highest_losses):
            # Convert image tensor to numpy array if necessary
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy().squeeze()  # Assuming single-channel

            # Add a row to the table
            table.add_data(
                wandb.Image(img, caption=f"Pred: {int(pred)}, True: {int(true_label)}"),
                int(pred),
                int(true_label),
                float(loss_val)
            )

        # Log the table
        wandb.log({"High-Loss Examples Table": table})

        print("Evaluation complete. Metrics and hardest examples logged as a table.")

def evaluate(model, test_loader, config, k=32):
    """
    Evaluate the model on the test dataset and identify hardest examples.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): Test DataLoader.
        config (dict): Configuration dictionary.
        k (int): Number of hardest examples to retrieve.

    Returns:
        tuple: (average loss, accuracy, highest losses, hardest examples, true labels, predictions)
    """
    model.eval()
    criterion = FocalLoss(
        alpha=config.get('model_params', {}).get('alpha', None),  # Adjust if using a different alpha
        gamma=config['train_params']['focal_gamma'],
        logits=True,
        reduce=True
    )

    test_loss = 0.0
    preds_list, labels_list, losses = [], [], []
    hardest_examples = []
    true_labels = []
    predictions = []

    for images, labels in tqdm(test_loader, desc='Evaluating', leave=False, unit='batch'):
        images = images.to(config['device'], non_blocking=True)
        labels = labels.to(config['device']).unsqueeze(1).float()

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(outputs)
        preds = (probabilities >= 0.5).float()

        losses.extend(loss.detach().cpu().numpy())
        predictions.extend(preds.detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    accuracy = (np.array(predictions) == np.array(labels_list)).mean() * 100

    # Identify hardest k examples
    losses = np.array(losses)
    hardest_indices = losses.argsort()[-k:]
    hardest_examples = [test_loader.dataset[i][0] for i in hardest_indices]
    true_labels = [test_loader.dataset[i][1] for i in hardest_indices]
    predictions = [predictions[i][0] for i in hardest_indices]

    return avg_test_loss, accuracy, losses[hardest_indices], hardest_examples, true_labels, predictions
