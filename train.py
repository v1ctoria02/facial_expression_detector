import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import config as cfg

LOGGER = logging.getLogger(__name__)


def load_images_from_folder(path: str, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomCrop((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "validation")
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model: nn.Module, input_loader: DataLoader, epochs: int = 3, use_gpu: bool = False):
    # Create NeuralNetwork object
    num_of_params = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Number of NN parameters: %s", num_of_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        criterion.cuda()

    losses = []
    for e in range(epochs):
        model.train()
        for batch_idx, [images_batch, labels_batch] in tqdm(enumerate(input_loader), "Batch progress"):
            if use_gpu:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()
            LOGGER.debug("Batch size: %d", len(images_batch))

            # The optimizer knows about all model parameters. These in turn store their own gradients.
            # When calling loss.backward() the newly computed gradients are added on top of the existing ones.
            # Thus at before calculating new gradients we need to clear the old ones using the zero_grad() method.
            optimizer.zero_grad()
            # Compute the forward pass of the model.
            output = model(images_batch)
            LOGGER.debug("Output size: %d", len(output))
            # Compute the loss between the prediction and the label
            loss = criterion(output, labels_batch)
            # Here pytorch applies backpropagation for us
            loss.backward()
            # Add the gradients onto the model parameters as specified by the optimizer and the learning rate
            optimizer.step()

            # Record the loss
            losses.append(loss.item())

        LOGGER.info("Average loss after %d epoch: %.4f", e, sum(losses) / len(losses))


def evaluate(model: nn.Module, test_loader: DataLoader, use_gpu: bool = True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images_batch, labels_batch in test_loader:
            if use_gpu:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(images_batch)
            _, predicted = torch.max(output.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    accuracy = 100 * correct / total
    LOGGER.info(f"Accuracy on the test set: {accuracy:.2f}%")
