# dsets/adni.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class ADNI(Dataset):
    _train_patient_ids = None
    _test_patient_ids = None
    full_df = None  # To store the complete DataFrame for splitting

    # Binary label mapping
    class_to_binary_label = {
        'AD': 1,
        'CN': 0,
        'MCI': 0
    }

    def __init__(self, config, split='train', test_size=0.2, random_seed=42):
        """
        Initializes the ADNI dataset.

        Args:
            config (dict): Configuration dictionary.
            split (str): One of 'train', 'test', or 'all'.
            test_size (float): Fraction of data to use for testing.
            random_seed (int): Seed for reproducibility.
        """
        self.path = os.path.join(config['project_root'], config['datasets']['adni']['data_dir'])
        self.split = split
        self.transform = config['transform']
        print(f'Using transform: {self.transform}')

        # Create the reference DataFrame
        if ADNI.full_df is None:
            ADNI.full_df = self.create_ref_df(self.path)

        # Perform splitting based on patient IDs
        if split in ['train', 'test']:
            self._prepare_split(test_size, random_seed)
            if split == 'train':
                self.df = ADNI.full_df[ADNI.full_df['patient_id'].isin(ADNI._train_patient_ids)].reset_index(drop=True)
            else:
                self.df = ADNI.full_df[ADNI.full_df['patient_id'].isin(ADNI._test_patient_ids)].reset_index(drop=True)
        elif split == 'all':
            self.df = ADNI.full_df.copy()
        else:
            raise ValueError("split must be one of 'train', 'test', or 'all'")

        # Extract image paths and labels
        self.image_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        # Define binary labels
        self.binary_labels = [self.class_to_binary_label[label] for label in self.labels]

    def _prepare_split(self, test_size, random_seed):
        """
        Splits the patient IDs into train and test sets and stores them as class variables.
        Stratifies based on per-patient binary labels.

        Args:
            test_size (float): Fraction of data to use for testing.
            random_seed (int): Seed for reproducibility.
        """
        if ADNI._train_patient_ids is None or ADNI._test_patient_ids is None:
            # Extract unique patient IDs and their corresponding labels
            df_patient = ADNI.full_df.drop_duplicates('patient_id')
            patient_ids = df_patient['patient_id'].values
            # Map textual labels to binary labels
            labels_per_patient = [self.class_to_binary_label[label] for label in df_patient['label']]

            # Perform the split with stratification to maintain class distribution
            train_ids, test_ids = train_test_split(
                patient_ids,
                test_size=test_size,
                random_state=random_seed,
                stratify=labels_per_patient
            )
            ADNI._train_patient_ids = train_ids
            ADNI._test_patient_ids = test_ids

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
        classes = ['AD', 'CN', 'MCI']
        for cls_label in classes:
            cls_dir = os.path.join(dataset_path, cls_label)
            if not os.path.isdir(cls_dir):
                print(f"Directory {cls_dir} does not exist. Skipping.")
                continue
            for subject in os.listdir(cls_dir):
                subject_dir = os.path.join(cls_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue
                # Assuming each subject has a unique identifier (e.g., Subject_01)
                patient_id = subject  # Modify this if your subject folders have a different naming convention
                for scan in os.listdir(subject_dir):
                    scan_dir = os.path.join(subject_dir, scan)
                    if not os.path.isdir(scan_dir):
                        continue
                    for file in os.listdir(scan_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(scan_dir, file)
                            paths.append(image_path)
                            labels.append(cls_label)
                            patient_ids.append(patient_id)
        full_df = pd.DataFrame({
            'path': paths,
            'label': labels,
            'patient_id': patient_ids
        })
        print(f"Total samples in ADNI dataset: {len(full_df)}")
        return full_df

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

