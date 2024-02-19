"""
This file contains a simple CNN model for the embeddings. The model is a simple CNN model with 3 convolutional layers and
1 fully connected layer. The model is defined using the PyTorch library. The model is defined in the CNN class. The
forward function is defined to define the forward pass of the model. The model is defined in the device specified in the
__init__.py file. The device can be 'cpu', 'cuda', or 'mps'. See __init__.py for more details.

The model is defined as follows:
1. The first convolutional layer has 3 input channels, 32 output channels, 3 kernel size, 1 stride, and 0 padding.
2. The second convolutional layer has 32 input channels, 64 output channels, 3 kernel size, 1 stride, and 0 padding.
3. The third convolutional layer has 64 input channels, 128 output channels, 3 kernel size, 1 stride, and 0 padding.
4. The pooling layer is a max pooling layer with 2 kernel size and 2 stride.
5. The fully connected layer has 128 * 26 * 26 input features and 1024 output features.

Classes:
    CNN: Defines the structure of the CNN model and the forward pass.

Functions:
    forward(self, x): Defines the forward pass of the model.

NOTE:
THIS IS A SIMPLE AND EXPERIMENTAL CNN MODEL FOR THE EMBEDDINGS. DON'T USE THIS MODEL FOR PRODUCTION. USE A BETTER MODEL
FOR PRODUCTION.
"""

# Importing the required libraries
import torch.nn as nn
import torch.nn.functional as F
from . import device


# Defining the CNN model
class CNN(nn.Module):
    """
    A simple convolutional neural network (CNN) for generating embeddings from frames.

    Methods:
        forward(self, x): Defines the forward pass of the model.
    """

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
