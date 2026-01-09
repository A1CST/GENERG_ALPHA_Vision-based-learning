import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__()
        # Input: 100x100 grayscale
        
        # Block 1: Detect basic edges/lines
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Detect simple shapes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Detect complex letter parts
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2) # Reduces size by half
        self.silu = nn.SiLU()
        
        # Calculate Flatten Size: 
        # 100x100 -> Pool -> 50x50 -> Pool -> 25x25 -> Pool -> 12x12
        # 12 * 12 * 128 filters = 18432
        self.fc1 = nn.Linear(12 * 12 * 128, 256)
        self.fc2 = nn.Linear(256, 26)

    def forward(self, x):
        # 1. Unflatten: (Batch, 10000) -> (Batch, 1, 100, 100)
        x = x.view(-1, 1, 100, 100)
        
        x = self.pool(self.silu(self.bn1(self.conv1(x)))) # -> 50x50
        x = self.pool(self.silu(self.bn2(self.conv2(x)))) # -> 25x25
        x = self.pool(self.silu(self.bn3(self.conv3(x)))) # -> 12x12
        
        x = torch.flatten(x, 1) # Flatten for dense layers
        x = self.silu(self.fc1(x))
        x = self.fc2(x)
        return x