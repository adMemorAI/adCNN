import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from pathlib import Path

from config import transform  

class OASISKaggle(Dataset):
    _train_patient_ids = None
    _test_patient_ids = None

    # Define class to binary label mapping as a class variable
    class_to_binary_label = {
        'no-dementia': 0,
        'verymild-dementia': 1,
        'mild-dementia': 1,
        'moderate-dementia': 1
    }

    @classmethod
    def _prepare_split(cls, df, test_size=0.2, random_seed=42):
        """
        Splits the patient IDs into train and test sets and stores them as class variables.
        Stratifies based on per-patient binary labels.
        """
        if cls._train_patient_ids is None or cls._test_patient_ids is None:
            # Extract unique patient IDs and their corresponding labels
            df_patient = df.drop_duplicates('patient_id')
            patient_ids = df_patient['patient_id'].values
            # Map textual labels to binary labels
            labels_per_patient = [cls.class_to_binary_label[label] for label in df_patient['label']]

            # Perform the split with stratification to maintain class distribution
            train_ids, test_ids = train_test_split(
                patient_ids,
                test_size=test_size,
                random_state=random_seed,
                stratify=labels_per_patient
            )
            cls._train_patient_ids = train_ids
            cls._test_patient_ids = test_ids

    def __init__(self, split='train', transform=transform, test_size=0.2, random_seed=42):
        """
        Initializes the dataset.

        Args:
            split (str): One of 'train', 'test', or 'all'.
            transform (callable, optional): Transform to apply to images.
            test_size (float): Fraction of data to use for testing.
            random_seed (int): Seed for reproducibility.
        """
        project_root = Path(__file__).parent.parent.parent
        dataset_dir = os.path.join(project_root, "datasets") # expected path: /adCNN/datasets
        self.path = os.path.join(dataset_dir, "oasis_kaggle")
        self.split = split
        self.transform = transform

        # Create the reference DataFrame
        self.df = self.create_ref_df(self.path)

        # Perform splitting based on patient IDs
        if split in ['train', 'test']:
            OASISKaggle._prepare_split(self.df, test_size, random_seed)
            if split == 'train':
                self.df = self.df[self.df['patient_id'].isin(OASISKaggle._train_patient_ids)].reset_index(drop=True)
            else:
                self.df = self.df[self.df['patient_id'].isin(OASISKaggle._test_patient_ids)].reset_index(drop=True)
        elif split == 'all':
            pass  # Use the entire dataset
        else:
            raise ValueError("split must be one of 'train', 'test', or 'all'")

        # Extract image paths and labels
        self.image_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        # Define binary labels
        self.binary_labels = [self.class_to_binary_label[label] for label in self.labels]

    @staticmethod
    def get_info_from_filename(filename):
        """
        Extracts patient and scan information from the filename.

        Args:
            filename (str): The image filename.

        Returns:
            tuple: (patient_id, mr_id, scan_id, layer_id) or (None, None, None, None) if pattern doesn't match.
        """
        pattern = re.compile(r'OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+).jpg', re.IGNORECASE)
        match = pattern.match(filename)
        if match:
            patient_id = int(match.group(1))
            mr_id = int(match.group(2))
            scan_id = int(match.group(3))
            layer_id = int(match.group(4))
            return patient_id, mr_id, scan_id, layer_id
        else:
            print(f"Filename {filename} does not match pattern")
            return None, None, None, None

    @classmethod
    def create_ref_df(cls, dataset_path):
        """
        Creates a DataFrame with image paths, labels, and patient IDs.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            pd.DataFrame: DataFrame containing 'path', 'label', and 'patient_id'.
        """
        paths, labels, patient_ids = [], [], []
        # Define class directories
        classes = ['no-dementia', 'verymild-dementia', 'mild-dementia', 'moderate-dementia']
        for cls_label in classes:
            cls_dir = os.path.join(dataset_path, cls_label)
            if not os.path.isdir(cls_dir):
                print(f"Directory {cls_dir} does not exist. Skipping.")
                continue
            for file in os.listdir(cls_dir):
                if file.lower().endswith('.jpg'):
                    patient_id, mr_id, scan_id, layer_id = cls.get_info_from_filename(file)
                    if patient_id is None:
                        continue  # Skip files with invalid filenames
                    paths.append(os.path.join(cls_dir, file))
                    labels.append(cls_label)
                    patient_ids.append(patient_id)
        ref_df = pd.DataFrame({
            'path': paths,
            'label': labels,
            'patient_id': patient_ids
        })
        return ref_df

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (image, label)
        """
        image = Image.open(self.image_paths[idx]).convert('L')  # Load image in grayscale
        if self.transform:
            image = self.transform(image)
        label = self.binary_labels[idx]
        label = torch.tensor(label, dtype=torch.float32)  # BCEWithLogitsLoss expects float labels
        return image, label
