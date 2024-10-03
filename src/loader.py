import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.datasets import ImageFolder  # Example for another dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from dsets.oasis_kaggle import OASISKaggle

# Import your Dataset classes
# from your_module import OASISKaggle, OtherDatasetClass, ...

class DatasetLoader:
    def __init__(self, dataset_classes, test_size=0.2, random_seed=42, batch_size=16):
        """
        Initializes the DatasetLoader.

        Args:
            dataset_classes (list): A list of PyTorch Dataset classes (e.g., [OASISKaggle, ImageFolder, ...]).
            test_size (float): Proportion of the data to be used for testing.
            random_seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for data loading.
        """
        self.dataset_classes = dataset_classes
        self.test_size = test_size
        self.random_seed = random_seed
        self.batch_size = batch_size

    def load_datasets(self):
        """
        Loads and combines multiple datasets, splits them into train and test sets based on group if needed,
        and returns data loaders.

        Returns:
            train_loader (DataLoader): DataLoader for the training set.
            test_loader (DataLoader): DataLoader for the testing set.
        """
        combined_train_datasets = []
        combined_test_datasets = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for dataset_class in self.dataset_classes:
            print(f"Loading dataset: {dataset_class.__name__}")

            if dataset_class == OASISKaggle:
                # Instantiate train and test datasets separately
                train_dataset = dataset_class(split='train')
                test_dataset = dataset_class(split='test')

                combined_train_datasets.append(train_dataset)
                combined_test_datasets.append(test_dataset)
            else:
                # Handle other datasets (e.g., ImageFolder)
                # Assuming other Dataset classes handle their own paths and transforms
                dataset = dataset_class()  # Instantiate without arguments

                # Perform random split
                dataset_size = len(dataset)
                test_size_abs = int(self.test_size * dataset_size)
                train_size_abs = dataset_size - test_size_abs

                generator = torch.Generator().manual_seed(self.random_seed)
                train_subset, test_subset = random_split(dataset, [train_size_abs, test_size_abs], generator=generator)

                combined_train_datasets.append(train_subset)
                combined_test_datasets.append(test_subset)

        # Concatenate all training and testing datasets
        combined_train_dataset = ConcatDataset(combined_train_datasets)
        combined_test_dataset = ConcatDataset(combined_test_datasets)

        # Create DataLoaders
        train_loader = DataLoader(combined_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(combined_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader

