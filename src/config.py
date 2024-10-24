import torch
import os
from pathlib import Path
from torchvision import transforms

class Config:
    def __init__(self):
        # Training parameters
        self.num_epochs = 20
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.early_stopping_patience = 5

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Paths
        self.log_dir = 'runs/experiment'
        self.model_dir = 'models'
        self.project_root = Path(__file__).parent.parent

        # Data parameters
        self.num_workers = 4
        self.pin_memory = True

        # Loss function parameters
        self.focal_alpha = 1
        self.focal_gamma = 2

        # Scheduler parameters
        self.scheduler_factor = 0.1
        self.scheduler_patience = 2

        # Data transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Model configuration
        self.model_name = 'ResAD'  # Specify the model name here
        self.model_params = {
            "ResAD": {
                'input_channels': 1,  # Example parameter
                'num_classes': 1      # Binary classification
            }
        }
