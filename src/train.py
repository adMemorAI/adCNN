import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from dsets.oasis_kaggle import OASISKaggle
from models.resad import ResAD
from losses.focal_loss import FocalLoss
from utils.logger import Logger
from utils.saver import ModelSaver
from utils.metrics_handler import MetricsHandler

from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import torch.backends.cudnn as cudnn

def main():
    # Initialize configuration
    config = Config()

    # Setup logger
    logger = Logger("rich")

    # Initialize Model Saver
    saver = ModelSaver(experiment_name="ResAD_DementiaDetection", model_class_name="ResAD", model_dir=config.model_dir)

    # Initialize Metrics Handler
    metrics_handler = MetricsHandler(experiment_dir=saver.get_experiment_dir(), log_dir=config.log_dir)

    # Configure cuDNN
    cudnn.benchmark = True  # Enable benchmark mode for optimized performance

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
    logger.info(f"Class Weights: {class_weights}")

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
    logger.info(f'Model on device: {config.device}')

    # Loss function and optimizer
    criterion = FocalLoss(alpha=class_weights[1].item(), gamma=config.focal_gamma, logits=True, reduce=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True
    )

    # Metrics
    accuracy_metric = Accuracy(task="binary").to(config.device)
    precision_metric = Precision(task="binary").to(config.device)
    recall_metric = Recall(task="binary").to(config.device)
    f1_metric = F1Score(task="binary").to(config.device)

    # Early stopping variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{config.num_epochs}')
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

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        logger.info(f'Training Loss: {epoch_loss:.4f}')

        # Validation
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

                val_loader_tqdm.set_postfix(loss=loss.item())

        val_loss = val_running_loss / len(val_dataset)

        preds_tensor = torch.cat(preds_list)
        labels_tensor = torch.cat(labels_list)

        val_accuracy = accuracy_metric(preds_tensor, labels_tensor) * 100
        val_precision = precision_metric(preds_tensor, labels_tensor) * 100
        val_recall = recall_metric(preds_tensor, labels_tensor) * 100
        val_f1 = f1_metric(preds_tensor, labels_tensor) * 100

        logger.info(f'Validation Loss: {val_loss:.4f}, '
                    f'Accuracy: {val_accuracy:.2f}%, '
                    f'Precision: {val_precision:.2f}%, '
                    f'Recall: {val_recall:.2f}%, '
                    f'F1-Score: {val_f1:.2f}%')

        # Prepare metrics dictionary
        metrics = {
            "Validation Loss": round(val_loss, 4),
            "Accuracy": round(val_accuracy.item(), 2),
            "Precision": round(val_precision.item(), 2),
            "Recall": round(val_recall.item(), 2),
            "F1-Score": round(val_f1.item(), 2)
        }

        # Handle metrics logging
        metrics_handler.handle_metrics(metrics_dict=metrics, epoch=epoch+1)

        # Reset metrics
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save the best model
            saved_model_path = saver.save_model(model, epoch+1, val_loss, val_f1, best=True)
            logger.info(f"Validation loss decreased, saving new best model as {saved_model_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= config.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    # Load the best model weights after training
    model.load_state_dict(best_model_wts)
    final_model_filename = f"{saver.model_class_name}_final_bestLoss{best_loss:.4f}.pth"
    final_model_path = os.path.join(saver.experiment_dir, final_model_filename)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved to {final_model_path}')

    # Close Metrics Handler (closes TensorBoard writer)
    metrics_handler.close()

if __name__ == '__main__':
    main()