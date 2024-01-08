import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Simple CNN model"""

    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(14*14*64, 1024),
            nn.Linear(1024, 10)
        )
        self.fc1 = nn.Linear(28*28*32, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # Add the channel dim (unexistent for single channel images)
        x = x.unsqueeze(dim=-3)
        x = self.convnet(x)
        #x = torch.flatten(x)
        #x = self.fc1(x)
        #x = self.fc2(x)
        # Softmax makes all class scores sum to 1, making them probabilities
        # Unnecessary if we use argmax, but useful for other applications
        x = torch.log_softmax(x,dim=-1)
        return x