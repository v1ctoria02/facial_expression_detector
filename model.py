import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

LOGGER = logging.getLogger(__name__)


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
        # (48 - 3 + 2) // 2 + 1 = 24
        # 48x48 -> 24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1, stride=2)
        # 24x24 -> 12x12
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 12x12 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 8x8 -> 8x8
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(1, 1), padding=0, stride=1)

        # fully connected block
        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        # (batch_size, 32, 48, 48)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # (batch_size, 64, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # (batch_size, 128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        # Dropout layer with probability 0.5
        self.dropout = nn.Dropout(p=0.5)
        # (batch_size, 512)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        # (batch_size, output_size)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        # Change the first layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Change the last layer to output 6 classes
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class emoNet(nn.Module):
    network_config = [32, 32, "M", 64, 64, "M", 128, 128, "M"]

    def __init__(self, num_of_channels, num_of_classes):
        super(emoNet, self).__init__()
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 128, 64), nn.ELU(True), nn.Dropout(p=0.5), nn.Linear(64, num_of_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=True)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ELU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class BestNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.fc1 = nn.Sequential(nn.Linear(4608, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25))

        self.fc2 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.25))

        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = x.view(x.size(0), -1)  # Flatten layer
        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
