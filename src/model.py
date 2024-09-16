import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # fully connected
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1) # binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # output: [batch_size, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x))) # output: [batch_size, 64, 56, 56]
        x = x.view(-1, 64 * 56 * 56) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
