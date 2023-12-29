import logging

import numpy as np
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # output_size = (input_size - kernel_size + 2 * padding) // stride + 1
        # (48 - 3 + 2) // 2 + 1 = 24
        # 48x48 -> 24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1, stride=2)
        # 24x24 -> 12x12
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 12x12 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 8x8 -> 4x4
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)

        # fully connected block
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        LOGGER.debug("Shape after conv1: %s", x.shape)
        x = self.pool(x)
        LOGGER.debug("Shape after pool: %s", x.shape)
        x = self.relu(self.conv2(x))
        LOGGER.debug("Shape after conv2: %s", x.shape)
        x = self.pool2(x)
        LOGGER.debug("Shape after pool: %s", x.shape)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)
        LOGGER.debug("Shape after reshape: %s", x.shape)

        x = self.relu(self.fc1(x))
        LOGGER.debug("Shape after fc1: %s", x.shape)
        x = self.relu(self.fc2(x))
        LOGGER.debug("Shape after fc2: %s", x.shape)
        x = self.fc3(x)
        LOGGER.debug("Shape after fc3: %s", x.shape)
        return x
