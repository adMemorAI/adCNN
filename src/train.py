import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model import SimpleCNN
from torchvision import transforms
import os

from config import device, transform

train_dir = 'data/train'
test_dir = 'data/test'

train_dataset = MRIDataset(train_dir, transform=transform)
test_dataset = MRIDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.00

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/adCNN.pth')
print('Model saved to models/adCNN.pth')

