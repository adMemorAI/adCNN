# visualize_transformations.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from utils.config import load_config
from utils.data_loader import get_data_loaders
from PIL import Image
import numpy as np

def visualize_samples(config, output_folder='transformed_images', num_batches=1):
    """
    Save a batch of images after all transformations to an output folder.

    Args:
        config (dict): Configuration dictionary.
        output_folder (str): Path to the folder where images will be saved.
        num_batches (int): Number of batches to save.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get data loaders (we'll use the training loader)
    train_loader, _, _ = get_data_loaders(config)

    # Initialize a counter for image filenames
    image_counter = 0

    # Get the mean and std for unnormalization (if necessary)
    normalize = None
    transform_list = config['transform'].transforms
    for t in transform_list:
        if t.__class__.__name__ == 'Normalize':
            normalize = t
            break

    mean = torch.tensor(normalize.mean) if normalize else None
    std = torch.tensor(normalize.std) if normalize else None

    # Iterate over the specified number of batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        # Move images to CPU if they are on GPU
        images = images.cpu()
        labels = labels.cpu()

        # Unnormalize images if normalization was applied
        if normalize is not None:
            images = images * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
            images = images.clamp(0, 1)

        # Handle grayscale images
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # Convert to 3-channel images for saving

        # Save each image in the batch
        for i in range(images.size(0)):
            image = images[i]
            label = labels[i].item()
            # Convert tensor to PIL Image
            pil_image = transforms.ToPILImage()(image)
            # Construct filename
            filename = f'image_{image_counter}_label_{label}.png'
            filepath = os.path.join(output_folder, filename)
            # Save image
            pil_image.save(filepath)
            image_counter += 1

    print(f"Saved {image_counter} images to '{output_folder}'.")

if __name__ == "__main__":
    # Load the configuration
    config = load_config('../config.yaml')

    # Import necessary libraries for transformations
    from torchvision import transforms

    # Visualize samples and save images to the output folder
    visualize_samples(config, output_folder='transformed_images', num_batches=5)

