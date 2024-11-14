# evaluate_model.py

import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import heapq

from dsets.oasis_kaggle import OASISKaggle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from utils.config import load_config
from utils.data_loader import get_data_loaders
from dsets.dataset_factory import get_dataset
from utils.metrics import get_metrics
from models.model_factory import get_model
from losses.focal_loss import FocalLoss
from utils.model_utils import get_model_path

logger = logging.getLogger(__name__)

def evaluate_model(config):
    """
    Evaluate the trained model on the test dataset and log the k hardest examples of each class to a W&B table.
    """
    # Initialize W&B run if not already active
    if wandb.run is None:
        wandb.init(project=config.get('wandb_project', 'adCNN'), job_type="evaluate", config=config)
        run = wandb.run
        should_finish = True
    else:
        run = wandb.run
        should_finish = False

    try:
        # Load the trained model artifact
        trained_model_artifact = run.use_artifact("trained_model:latest", type="model")
        model_dir = trained_model_artifact.download()
        model_path = get_model_path(model_dir, pattern="trained_model.pth")

        if not model_path:
            logger.error("Trained model file not found in the artifact.")
            raise FileNotFoundError("Trained model file not found in the artifact.")

        # Initialize the model
        model = get_model(config['model']['type'], **config['model']['params'])
        model.load_state_dict(torch.load(model_path, map_location=config['device'], weights_only=True))
        model = model.to(config['device'])
        model.eval()

        # Load the test dataset
        test_dataset = get_dataset(
            dataset_type=config['datasets']['type'],
            split='test',
            transform=config['transform']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['train_params']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        # Log class distribution in the test set
        labels = test_dataset.binary_labels
        class_counts = np.bincount(labels)
        print(f"Test Set Class Distribution: {class_counts}")
        run.summary.update({
            "Test_No_Dementia": int(class_counts[0]),
            "Test_Dementia": int(class_counts[1])
        })

        # Define the loss function with per-sample loss
        criterion = FocalLoss(
            alpha=config.get('model', {}).get('params', {}).get('alpha', None),
            gamma=config['train_params']['focal_gamma'],
            logits=True,
        )

        # Define metrics
        metrics = get_metrics(config['device'])

        # Initialize W&B Table for Hardest Test Examples
        table = wandb.Table(columns=["ID", "Image", "Prediction", "True Label", "Loss"])

        # Initialize tracking variables
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_true_labels = []

        # Retrieve the number of hardest examples to log per class
        k = config.get('evaluate_params', {}).get('k_hard_examples', 10)
        hardest_examples = {0: [], 1: []}  # Assuming binary classification

        # Evaluate the model on the test set
        with torch.no_grad():
            for batch_idx, (images, labels_batch) in enumerate(tqdm(test_loader, desc='Evaluating', leave=False, unit='batch')):
                images = images.to(config['device'], non_blocking=True)
                labels_batch = labels_batch.to(config['device']).unsqueeze(1).float()

                outputs = model(images)
                per_sample_loss = criterion(outputs, labels_batch).detach().cpu().numpy().flatten()
                loss = per_sample_loss.mean()  # Mean loss for the batch
                probabilities = torch.sigmoid(outputs)
                preds = (probabilities >= 0.5).float()

                # Convert predictions and labels to integers and extend the lists
                all_predictions.extend(preds.cpu().int().numpy().flatten().tolist())
                all_true_labels.extend(labels_batch.cpu().int().numpy().flatten().tolist())

                # Update tracking variables
                total_loss += loss.item() * images.size(0)  # Total loss accumulates batch loss multiplied by batch size
                correct_predictions += (preds == labels_batch).sum().item()
                total_samples += images.size(0)

                # Iterate through each sample in the batch to identify hardest examples
                for i in range(images.size(0)):
                    if i >= len(per_sample_loss):
                        logger.warning(f"Batch {batch_idx}: Attempting to access per_sample_loss[{i}] with size {len(per_sample_loss)}")
                        continue  # Skip this sample

                    true_label = int(labels_batch[i].cpu().numpy())
                    loss_val = float(per_sample_loss[i])
                    pred = int(preds[i].cpu().numpy())

                    # Use a min-heap to keep top k hardest (highest loss) examples per class
                    if len(hardest_examples[true_label]) < k:
                        heapq.heappush(hardest_examples[true_label], (loss_val, batch_idx, i, images[i].cpu().numpy(), pred))
                    else:
                        if loss_val > hardest_examples[true_label][0][0]:
                            heapq.heappushpop(hardest_examples[true_label], (loss_val, batch_idx, i, images[i].cpu().numpy(), pred))

        # After evaluation, collect and sort the hardest examples for each class
        for class_label, examples in hardest_examples.items():
            # Sort examples by loss in descending order
            sorted_examples = sorted(examples, key=lambda x: x[0], reverse=True)
            for example in sorted_examples:
                loss_val, batch_idx, i, img, pred = example
                true_label = class_label
                row_id = f"{class_label}_{batch_idx}_{i}"

                # Process the image (assuming single-channel)
                img_processed = img.squeeze()
                table.add_data(
                    row_id,
                    wandb.Image(img_processed, caption=f"Pred: {pred}, True: {true_label}"),
                    pred,
                    true_label,
                    loss_val
                )

        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = (correct_predictions / total_samples) * 100

        # Calculate additional metrics
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        precision = precision_score(all_true_labels, all_predictions, average='weighted')
        recall = recall_score(all_true_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)


        unique_preds = set(all_predictions)
        unique_true = set(all_true_labels)

        wandb.log({
            "Unique Predictions": wandb.Histogram(list(unique_preds)),
            "Unique True Labels": wandb.Histogram(list(unique_true)),
        })

        # Check if number of unique predictions exceeds number of class names
        if len(unique_preds) > len(["No Dementia", "Dementia"]):
            logger.error(f"Number of unique predictions ({len(unique_preds)}) exceeds number of class names (2). Skipping confusion matrix.")
        else:
            # Log the metrics to W&B
            wandb.log({
                "Average Loss": avg_loss,
                "Accuracy (%)": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_true_labels,
                    preds=all_predictions,
                    class_names=["No Dementia", "Dementia"]
                ),
                "Hardest Test Examples Table": table
            })

        print("Evaluation complete. Metrics and hardest test examples logged to W&B.")

    finally:
        # Finish the W&B run if it was started here
        if should_finish:
            wandb.finish()

if __name__ == "__main__":
    # Example usage
    config = load_config('path_to_config.yaml')  # Replace with your config path
    evaluate_model(config)

