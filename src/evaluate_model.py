# src/evaluate_model.py

import logging
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb  # Import W&B

from utils.config import load_config
from utils.data_loader import get_data_loaders
from utils.metrics import get_metrics
from models.model_factory import get_model
from losses.focal_loss import FocalLoss

logger = logging.getLogger(__name__)

def evaluate_model(config):
    """
    Evaluate the trained model on the validation dataset.
    """
    # Initialize W&B run
    wandb.init(project=config.get('wandb_project', 'adCNN'), job_type="evaluate")
    
    # Define table columns
    columns = ["id", "image", "guess", "truth", "score_0", "score_1"]
    
    # Initialize the table
    prediction_table = wandb.Table(columns=columns)
    
    # Set the maximum number of images to log
    max_images = 100  # Adjust based on your preference
    
    # Load the final model artifact
    try:
        model_artifact = wandb.use_artifact("final_model:latest", type="model")
        model_artifact_dir = model_artifact.download()
        model_path = os.path.join(model_artifact_dir, 'final_model.pth')
        logger.info(f"Model artifact downloaded to {model_path}")
    except wandb.CommError as e:
        logger.error(f"Error fetching artifact: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e

    # Initialize the model
    model = get_model(config['model']['type'], **config['model']['params'])
    model.load_state_dict(torch.load(model_path))
    model.to(config['device'])

    # Get data loaders and class weights
    _, val_loader, _ = get_data_loaders(config)

    # Define loss function and metrics
    criterion = FocalLoss(
        alpha=[1.0, 1.0],  # Binary classification: [alpha_non_dementia, alpha_dementia]
        gamma=config['evaluate_params']['focal_gamma'],
        logits=True,
        reduce=True
    )
    metrics = get_metrics(config['device'])  # Ensure this function is correctly implemented

    # Perform evaluation
    try:
        average_loss, average_accuracy, f1, precision, recall, conf_matrix = evaluate(
            model, val_loader, criterion, metrics, config['device'], prediction_table, max_images
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e

    # Log metrics and table to W&B
    wandb.log({
        "Evaluation Loss": average_loss,
        "Evaluation Accuracy": average_accuracy,
        "Evaluation F1-Score": f1,
        "Evaluation Precision": precision,
        "Evaluation Recall": recall,
        "Confusion Matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_labels,
            preds=predictions,
            class_names=["Non-Dementia", "Dementia"]
        ),
        "Predictions Table": prediction_table
    })

    logger.info("Evaluation complete. Metrics and predictions logged to W&B.")

    # Finish the W&B run
    wandb.finish()

def evaluate(model, dataloader, criterion, metrics, device, prediction_table=None, max_images=100):
    model.eval()
    losses = []
    accuracies = []
    predictions = []
    true_labels = []
    image_counter = 0  # To limit the number of images logged
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).float()  # Ensure labels are [batch_size, 1] and float
            
            outputs = model(inputs)  # [batch_size, 1]
            loss = criterion(outputs, labels)
            preds = (outputs > 0).float()  # Binary predictions (0 or 1)
            
            # Compute probabilities
            probs = torch.sigmoid(outputs)
            p1 = probs.squeeze(1)
            p0 = 1 - p1
            
            # Populate W&B table
            if prediction_table is not None and image_counter < max_images:
                images = inputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                p0_np = p0.cpu().numpy()
                p1_np = p1.cpu().numpy()
                
                for i in range(inputs.size(0)):
                    if image_counter >= max_images:
                        break
                    # Convert image for visualization
                    img = images[i].squeeze()  # [1, H, W] -> [H, W]
                    wandb_img = wandb.Image(img, caption=f"Truth: {int(labels_np[i][0])}, Pred: {int(preds_np[i][0])}")
                    
                    label = int(labels_np[i][0])
                    pred = int(preds_np[i][0])
                    score0 = float(p0_np[i])
                    score1 = float(p1_np[i])
                    row_id = f"{batch_idx}_{i}"
                    
                    # Add row to the table
                    prediction_table.add_data(row_id, wandb_img, pred, label, score0, score1)
                    image_counter += 1
            
            losses.append(loss.detach().cpu().numpy())
            accuracies.append(torch.sum(preds == labels).item() / labels.size(0))
            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())
    
    # Convert lists to numpy arrays
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate overall metrics
    average_loss = losses.mean()
    average_accuracy = accuracies.mean()
    f1 = f1_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    return average_loss, average_accuracy, f1, precision, recall, conf_matrix
