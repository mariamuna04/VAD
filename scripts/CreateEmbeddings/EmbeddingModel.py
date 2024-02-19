import torch.nn as nn
import torch.nn.functional as F
from . import device


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0, device=device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, device=device)
        self.d = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, device=device)
        # Pooling layer, All are same
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 26 * 26, 1024, device=device)  # Adjust input size based on your frame size

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # For batch size 1
        x = x.view(-1, 128 * 26 * 26)
        # For batch size > 1
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x
