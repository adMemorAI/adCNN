import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  
import subprocess

from configs.config import Config
from dsets.oasis_kaggle import OASISKaggle
from models.resad import ResAD
from losses.focal_loss import FocalLoss
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import torch.backends.cudnn as cudnn
import os

def get_git_commit():
    """Retrieve the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"

def main():
    # Initialize configuration
    config = Config()

    # Configure cuDNN for optimized performance
    cudnn.benchmark = True

    # Initialize wandb
    wandb.init(
        project="ResNet",  # Replace with your wandb project name
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "focal_gamma": config.focal_gamma,
            "scheduler_factor": config.scheduler_factor,
            "scheduler_patience": config.scheduler_patience,
            "optimizer": "Adam",
            "loss_function": "FocalLoss",
            "model": "ResAD",
            "git_commit": get_git_commit(),
            # Add other hyperparameters as needed
        },
        name=f"Run-{wandb.util.generate_id()}",
        tags=["training", "ResAD", "baseline"],  # Add relevant tags
    )

    # Log additional configurations if necessary
    wandb.config.update({
        "experiment_dir": config.model_dir,
        "log_dir": config.model_dir,  # Assuming log_dir is same as model_dir
    }, allow_val_change=True)

    # Load datasets
    train_dataset = OASISKaggle(split='train', transform=config.transform)
    val_dataset = OASISKaggle(split='test', transform=config.transform)

    # Calculate class weights for handling class imbalance
    train_labels = np.array(train_dataset.binary_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    wandb.log({"Class Weights": class_weights.tolist()})

    # Create weighted sampler for the training data loader
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Initialize model
    model = ResAD().to(config.device)

    # Watch the model (logs gradients and model graph)
    wandb.watch(model, log="all")

    # Define loss function and optimizer
    criterion = FocalLoss(
        alpha=class_weights[1].item(),
        gamma=config.focal_gamma,
        logits=True,
        reduce=True
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True
    )

    # Define metrics
    accuracy_metric = Accuracy(task="binary").to(config.device)
    precision_metric = Precision(task="binary").to(config.device)
    recall_metric = Recall(task="binary").to(config.device)
    f1_metric = F1Score(task="binary").to(config.device)

    # Initialize variables for early stopping and best model tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        wandb.log({"Epoch": epoch + 1})  # Log the current epoch

        # Training Phase
        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False, unit='batch')

        for images, labels in train_loader_tqdm:
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device).unsqueeze(1).float()  # Ensure labels have shape [batch_size, 1]

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Update tqdm progress bar
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        wandb.log({"Train Loss": epoch_loss})

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        preds_list = []
        labels_list = []
        probs_list = []

        val_loader_tqdm = tqdm(val_loader, desc='Validation', leave=False, unit='batch')

        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images = images.to(config.device, non_blocking=True)
                labels = labels.to(config.device).unsqueeze(1).float()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                probabilities = torch.sigmoid(outputs)
                preds = (probabilities >= 0.5).float()

                preds_list.append(preds)
                labels_list.append(labels)
                probs_list.append(probabilities)

                # Update tqdm progress bar
                val_loader_tqdm.set_postfix(loss=loss.item())

        val_loss = val_running_loss / len(val_dataset)

        preds_tensor = torch.cat(preds_list)
        labels_tensor = torch.cat(labels_list)

        val_accuracy = accuracy_metric(preds_tensor, labels_tensor) * 100
        val_precision = precision_metric(preds_tensor, labels_tensor) * 100
        val_recall = recall_metric(preds_tensor, labels_tensor) * 100
        val_f1 = f1_metric(preds_tensor, labels_tensor) * 100

        # Log validation metrics to wandb
        wandb.log({
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy.item(),
            "Validation Precision": val_precision.item(),
            "Validation Recall": val_recall.item(),
            "Validation F1-Score": val_f1.item(),
            "Learning Rate": optimizer.param_groups[0]['lr'],
        })

        # Reset metrics for next epoch
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Step the scheduler
        scheduler.step(val_loss)

        # Early Stopping and Best Model Saving
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save the best model
            saved_model_path = os.path.join(config.model_dir, f"ResAD_best_loss_{best_loss:.4f}.pth")
            torch.save(model.state_dict(), saved_model_path)
            wandb.save(saved_model_path)  # Log the model file

            # Log the model checkpoint as an artifact
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(saved_model_path)
            wandb.log_artifact(artifact)

            wandb.run.summary["best_val_loss"] = best_loss
            wandb.run.summary["best_val_f1"] = val_f1.item()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                wandb.log({"Early Stopping": True})
                break

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    final_model_path = os.path.join(config.model_dir, f"ResAD_final_bestLoss_{best_loss:.4f}.pth")
    torch.save(model.state_dict(), final_model_path)

    # Log the final model as an artifact
    artifact = wandb.Artifact('final_model', type='model')
    artifact.add_file(final_model_path)
    wandb.log_artifact(artifact)

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
