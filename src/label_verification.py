import matplotlib.pyplot as plt
import random
import numpy as np
import torch

from tqdm import tqdm

def imshow(img, title):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_samples(loader, num_samples=5):
    data_iter = iter(loader)
    try:
        images, labels = next(data_iter)
    except StopIteration:
        print("The loader is empty.")
        return
    for _ in range(num_samples):
        if images.size(0) == 0:
            print("No images in this batch.")
            break
        idx = random.randint(0, images.size(0) - 1)
        imshow(images[idx], f'Label: {labels[idx].item()}')

def get_label_distribution(loader):
    dataset = loader.dataset  # This is a ConcatDataset
    labels = []

    for subset in tqdm(dataset.datasets, desc="Computing Label Distribution", unit="subset"):
        subset_dataset = subset.dataset  # Original dataset (e.g., OASISKaggle)
        subset_indices = subset.indices  # Indices in the original dataset

        subset_labels = [subset_dataset.labels[idx] for idx in subset_indices]
        labels.extend(subset_labels)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    return distribution


def verify_label_values(loader):
    valid = True
    for _, labels in tqdm(loader, desc="Verifying Label Values", unit="batch"):
        if not torch.all((labels == 0) | (labels == 1)):
            valid = False
            break
    return valid

def check_label_types(loader):
    for _, labels in tqdm(loader, desc="Checking Label Types", unit="batch"):
        if labels.dtype not in [torch.float32, torch.float64]:
            print(f'Unexpected label type: {labels.dtype}')
            return False
    return True

def check_alignment(loader):
    for images, labels in tqdm(loader, desc="Checking Alignment", unit="batch"):
        if images.size(0) != labels.size(0):
            print("Mismatch between images and labels batch sizes.")
            return False
    return True

def perform_label_verification(train_loader, test_loader):
    print("=== Visual Inspection ===")
    print("Training Data Samples:")
    visualize_samples(train_loader, num_samples=5)
    print("Validation Data Samples:")
    visualize_samples(test_loader, num_samples=5)

    print("\n=== Label Distribution ===")
    train_distribution = get_label_distribution(train_loader)
    test_distribution = get_label_distribution(test_loader)
    print(f'Train Label Distribution: {train_distribution}')
    print(f'Test Label Distribution: {test_distribution}')

    print("\n=== Label Value Verification ===")
    train_labels_valid = verify_label_values(train_loader)
    test_labels_valid = verify_label_values(test_loader)
    print(f'Training Labels Valid: {train_labels_valid}')
    print(f'Validation Labels Valid: {test_labels_valid}')

    print("\n=== Label Type Verification ===")
    train_label_types = check_label_types(train_loader)
    test_label_types = check_label_types(test_loader)
    print(f'Training Labels Float: {train_label_types}')
    print(f'Validation Labels Float: {test_label_types}')

    print("\n=== Alignment Verification ===")
    alignment_train = check_alignment(train_loader)
    alignment_test = check_alignment(test_loader)
    print(f'Training Alignment Correct: {alignment_train}')
    print(f'Validation Alignment Correct: {alignment_test}')

