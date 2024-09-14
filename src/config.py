import torch
from torchvision import transforms

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
