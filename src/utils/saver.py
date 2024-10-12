import os
import torch
from datetime import datetime

class ModelSaver:
    def __init__(self, experiment_name, model_class_name, model_dir='models'):
        """
        Initializes the ModelSaver.

        Args:
            experiment_name (str): Name of the experiment.
            model_class_name (str): Name of the model class.
            model_dir (str): Directory where models will be saved.
        """
        self.experiment_name = experiment_name
        self.model_class_name = model_class_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(model_dir, f"{self.experiment_name}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def save_model(self, model, epoch, val_loss, val_f1, best=False):
        """
        Saves the model state_dict with a dynamic filename.

        Args:
            model (torch.nn.Module): The model to save.
            epoch (int): Current epoch number.
            val_loss (float): Validation loss.
            val_f1 (float): Validation F1-score.
            best (bool): Flag indicating if this is the best model so far based on val_loss.

        Returns:
            str: Path to the saved model file.
        """
        if best:
            filename = f"{self.model_class_name}_best_epoch{epoch}_valLoss{val_loss:.4f}_f1{val_f1:.2f}.pth"
        else:
            filename = f"{self.model_class_name}_epoch{epoch}_valLoss{val_loss:.4f}_f1{val_f1:.2f}_{self.timestamp}.pth"
        filepath = os.path.join(self.experiment_dir, filename)
        torch.save(model.state_dict(), filepath)
        return filepath
    
    def get_experiment_dir(self):
        """
        Returns the directory of the current experiment.

        Returns:
            str: Path to the experiment directory.
        """
        return self.experiment_dir