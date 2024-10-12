import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResAD(nn.Module):
    def __init__(self):
        super(SCNN4, self).__init__()
        # Load the pre-trained ResNet18 model with the new weights parameter
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer to accept 1-channel images
        self.base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Initialize the new conv1 layer
        nn.init.kaiming_normal_(self.base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Modify the output layer for binary classification
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.base_model(x)
        return x