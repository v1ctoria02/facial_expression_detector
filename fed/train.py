import logging
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import fed.config as cfg

_logger = logging.getLogger(__name__)


def load_images_from_folder(path: str, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "validation")
    if not os.path.exists(test_path):
        test_path = os.path.join(path, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class EarlyStop:
    """Early stopping callback for the training loop."""

    def __init__(self, patience: int = 5, less_is_better: bool = True):
        self.patience = patience
        self.less_is_better = less_is_better
        self.best_value = float("inf") if less_is_better else float("-inf")
        self.patience_counter = 0
        self.best_model = None

    def __call__(self, model: nn.Module, value: float) -> bool:
        """Check if the validation loss is not improving."""
        stop_check = (value < self.best_value) if self.less_is_better else (value > self.best_value)
        if stop_check:
            self.best_value = value
            self.patience_counter = 0
            self.best_model = model.state_dict()
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            model.load_state_dict(self.best_model)
            return True
        return False


def train(
    model: nn.Module, input_loader: DataLoader, valid_loader: DataLoader = None, epochs: int = 3, use_gpu: bool = False
) -> dict[str, list[float]]:
    """Train the model on the training set and evaluate it on the validation set."""
    num_of_params = sum(parameter.numel() for parameter in model.parameters())
    _logger.info(f"Number of NN parameters: {num_of_params:_}".replace("_", "."))

    # Create the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        criterion.cuda()

    # Learning rate scheduler - learning rate will decrease by a factor of gamma every step_size epochs
    lr_scheduler = StepLR(optimizer, step_size=cfg.LEARNING_RATE_DECAY_STEP, gamma=cfg.LEARNING_RATE_DECAY)
    early_stop = EarlyStop()

    _logger.info(
        "Initializing `%s` training on %d epochs with `%s` optimizer with learning rate %.4f. "
        "`%s` LRscheduler step size %d and %.4f decay.",
        type(model).__name__,
        cfg.EPOCHS,
        type(optimizer).__name__,
        cfg.LEARNING_RATE,
        type(lr_scheduler).__name__,
        cfg.LEARNING_RATE_DECAY_STEP,
        cfg.LEARNING_RATE_DECAY,
    )

    stats = {"train_loss": [], "valid_loss": [], "accuracy": []}
    progress_bar = tqdm(range(epochs), "Training progress [epochs]", total=epochs, leave=False)
    for e in progress_bar:
        model.train()
        for images_batch, labels_batch in input_loader:
            if use_gpu:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            # The optimizer knows about all model parameters. These in turn store their own gradients.
            # When calling loss.backward() the newly computed gradients are added on top of the existing ones.
            # Thus before calculating new gradients we need to clear the old ones using the zero_grad() method.
            optimizer.zero_grad()
            # Compute the forward pass of the model.
            output = model(images_batch)
            # Compute the loss between the prediction and the label
            loss = criterion(output, labels_batch)
            # PyTorch applies backpropagation
            loss.backward()
            # Add the gradients onto the model parameters as specified by the optimizer and the learning rate
            optimizer.step()
            # Record the loss
            stats["train_loss"].append(loss.item())

        training_loss = sum(stats["train_loss"]) / len(stats["train_loss"])
        if valid_loader is not None:
            # Evaluate the model on the validation set if it is provided
            validation_loss, accuracy = evaluate(model, valid_loader, use_gpu)
            stats["valid_loss"].append(validation_loss)
            stats["accuracy"].append(accuracy)
            progress_bar.set_postfix({
                "Accuracy": accuracy,
                "Train loss": training_loss,
                "Valid loss": sum(stats["valid_loss"]) / len(stats["valid_loss"]),
            })
            if early_stop(model, validation_loss):
                # Early stop if the validation loss is not improving
                _logger.info("Early stopping after %d epochs", e + 1)
                break
        else:
            # If there is no validation set, just print the training loss
            progress_bar.set_postfix({"Training loss": training_loss})

        # Update the learning rate
        lr_scheduler.step()
    return stats


def evaluate(model: nn.Module, loader: DataLoader, use_gpu: bool = True) -> tuple[float, float]:
    """Evaluate the model on the test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    losses: list[float] = []
    accuracies: list[float] = []

    with torch.no_grad():
        for images_batch, labels_batch in loader:
            if use_gpu:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(images_batch)
            loss = criterion(output, labels_batch)
            # record the loss
            losses.append(loss.item())
            # Compute the accuracy
            _, predicted = torch.max(output.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            accuracies.append(100 * correct / total)
    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_loss, avg_accuracy
