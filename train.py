import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def load_images_from_folder(path: str, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "validation")
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(model: nn.Module, input_loader: DataLoader, epoch: int = 3, use_gpu: bool = False):
    # Create NeuralNetwork object
    num_of_params = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Number of NN parameters: %s", num_of_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    losses = []
    for e in range(epoch):
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

            if batch_idx % 100 == 0:
                LOGGER.info("Epoch: %d, Batch: %d, Loss: %.4f", epoch, batch_idx, loss.item())
        LOGGER.info("Average loss for %d epoch: %.4f", e, sum(losses)/len(losses))

    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

def evaluate(model: nn.Module, test_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    LOGGER.info(f'Accuracy on the test set: {accuracy:.2f}%')
