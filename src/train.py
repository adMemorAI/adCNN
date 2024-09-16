import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SimpleCNN  # Or your updated model
import os

from config import device, transform
from loader import DatasetLoader
from dsets.oasis_kaggle import OASISKaggle

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import Accuracy, Precision, Recall, F1Score

from label_verification import perform_label_verification

# Initialize rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("rich")

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/adCNN_experiment')

datasets_info = [
    (OASISKaggle, {"name": "oasis_kaggle"}),
]

loader = DatasetLoader(datasets_info, test_size=0.2, random_seed=42, batch_size=64)
train_loader, test_loader = loader.load_datasets()

# Perform Label Verification Before Training
#logger.info("Starting Label Verification...")
#perform_label_verification(train_loader, test_loader)
#logger.info("Label Verification Completed.\n")

model = SimpleCNN().to(device)
# Optionally apply weight initialization
# model.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjusted learning rate

# Initialize metrics
accuracy_metric = Accuracy(task="binary").to(device)
precision_metric = Precision(task="binary").to(device)
recall_metric = Recall(task="binary").to(device)
f1_metric = F1Score(task="binary").to(device)

# Device information
logger.info(f'Model on device: {next(model.parameters()).device}')

num_epochs = 10

for epoch in range(num_epochs):
    logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
    model.train()
    running_loss = 0.00

    train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False, unit='batch')

    for images, labels in train_loader_tqdm:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()  # Ensure labels are float

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        # Optionally clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        train_loader_tqdm.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    logger.info(f'Training Loss: {epoch_loss:.4f}')
    writer.add_scalar('Loss/Train', epoch_loss, epoch)

    # Validation
    model.eval()
    running_val_loss = 0.00
    correct = 0
    total = 0
    test_loader_tqdm = tqdm(test_loader, desc='Validation', leave=False)

    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            # Update metrics
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

            test_loader_tqdm.set_postfix(loss=loss.item())

    val_loss = running_val_loss / len(test_loader.dataset)
    val_accuracy = accuracy_metric.compute() * 100
    val_precision = precision_metric.compute() * 100
    val_recall = recall_metric.compute() * 100
    val_f1 = f1_metric.compute() * 100

    logger.info(f'Validation Loss: {val_loss:.4f}, '
                f'Accuracy: {val_accuracy:.2f}%, '
                f'Precision: {val_precision:.2f}%, '
                f'Recall: {val_recall:.2f}%, '
                f'F1-Score: {val_f1:.2f}%')

    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    writer.add_scalar('Precision/Validation', val_precision, epoch)
    writer.add_scalar('Recall/Validation', val_recall, epoch)
    writer.add_scalar('F1_Score/Validation', val_f1, epoch)

    # Reset metrics
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    # Summary Table
    table = Table(title=f"Epoch {epoch+1} Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Training Loss", f"{epoch_loss:.4f}")
    table.add_row("Validation Loss", f"{val_loss:.4f}")
    table.add_row("Validation Accuracy", f"{val_accuracy:.2f}%")
    table.add_row("Validation Precision", f"{val_precision:.2f}%")
    table.add_row("Validation Recall", f"{val_recall:.2f}%")
    table.add_row("Validation F1-Score", f"{val_f1:.2f}%")

    console.print(table)

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/adCNN.pth')
logger.info('Model saved to models/adCNN.pth')

writer.close()

