import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import transform

class OASISKaggle(Dataset):
    def __init__(self, path):
        """
        Args:
            path (string): Directory with all the images organized in class-specific subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.path = path 
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = ['no-dementia', 'verymild-dementia', 'mild-dementia', 'moderate-dementia']

        self.class_to_binary_label = {
            'no-dementia': 0,
            'verymild-dementia': 1,
            'mild-dementia': 1,
            'moderate-dementia': 1
        }

        for cls in self.classes:
            cls_dir = os.path.join(path, cls)
            for image_name in os.listdir(cls_dir):
                if image_name.endswith('.jpg'):
                    img_path = os.path.join(cls_dir, image_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_binary_label[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        return image, label
