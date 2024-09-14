import os
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

class DatasetLoader:
    def __init__(self, dataset_info, test_size=0.2, random_seed=42, batch_size=16):
        """
        Args:
            dataset_info (list): A list of tuples where each tuple contains:
                                    (dataset_class, dataset_args), where:
                                    - dataset_class is a PyTorch Dataset class (or custom dataset like OASISKaggle)
                                    - dataset_args is a dictionary of arguments to pass to the dataset class
                                  Example: 
                                  [
                                      (OASISKaggle, {"name": "oasis-kaggle", "transform": transform}),
                                      (ImageFolder, {"name": "another-dataset", "transform": transform})
                                  ]
            test_size (float): Proportion of the data to be used for testing.
            random_seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for data loading.
        """
        self.dataset_info = dataset_info
        self.test_size = test_size
        self.random_seed = random_seed
        self.batch_size = batch_size

    def load_datasets(self):
        """
        Loads and combines multiple datasets, splits them into train and test sets, and returns data loaders.
        
        Returns:
            train_loader: DataLoader for the training set.
            test_loader: DataLoader for the testing set.
        """
        combined_train_datasets = []
        combined_test_datasets = []

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for dataset_class, dataset_args in self.dataset_info:
            # Dynamically instantiate the dataset

            dataset_name = dataset_args.get("name", dataset_class.__name__)
            print(f"Loading dataset: {dataset_name}")
            dataset_path = os.path.join(project_root, "datasets", dataset_name)

            dataset = dataset_class(dataset_path)

            # Split the dataset into train and test sets
            dataset_size = len(dataset)
            test_size = int(self.test_size * dataset_size)
            train_size = dataset_size - test_size

            generator = torch.Generator().manual_seed(self.random_seed)
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

            combined_train_datasets.append(train_dataset)
            combined_test_datasets.append(test_dataset)

        combined_train_dataset = ConcatDataset(combined_train_datasets)
        combined_test_dataset = ConcatDataset(combined_test_datasets)

        train_loader = DataLoader(combined_train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(combined_test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

