import logging

from matplotlib import pyplot as plt

_logger = logging.getLogger(__name__)


def plot_stats(stats: dict[str, list[float]], save_path: str, final_accuracy: float) -> None:
    """Plot the training stats."""
    batches_per_epoch = int(len(stats["train_loss"]) / len(stats["valid_loss"]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(stats["train_loss"], label="Training Loss")
    valid_loss_x = range(0, len(stats["train_loss"]), batches_per_epoch)
    plt.plot(valid_loss_x, stats["valid_loss"], label="Validation Loss", linestyle="--", marker="o")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(stats["accuracy"])), stats["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("%")
    plt.title("Accuracy")
    plt.text(len(stats["accuracy"]) - 1, final_accuracy, f"{final_accuracy:.2f}%", ha="right", va="bottom")

    plt.tight_layout()  # Adjust spacing between subplots
    filename = f"{save_path}.png"
    _logger.info(filename)
    plt.savefig(filename)
