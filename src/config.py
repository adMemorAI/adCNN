import torch
from torchvision import transform

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image transform
transform = transform.Compose([
    transform.resize((224, 224)),
    transform.grayscale(),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5], std=[0.5])
])
