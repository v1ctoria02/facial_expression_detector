from matplotlib import pyplot as plt


def plot_stats(stats: dict[str, list[float]], model_path: str, final_accuracy: float) -> None:
    """Plot the training stats."""
    plt.figure(figsize=(12, 6))
    plt.title("Losses and Accuracy")
    plt.xlabel("Epochs")

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(stats["train_loss"], label="Average Training Loss")
    ax1.plot(stats["valid_loss"], label="Validation Loss", color="#097054")
    ax1.set_ylabel("Loss")

    ax2 = plt.twinx()
    ax2.set_ylabel("Accuracy %")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(range(0, 101, 10))
    ax2.plot(stats["accuracy"], label="Accuracy", color="orange", linestyle="--")
    ax2.text(len(stats["accuracy"]) - 1, final_accuracy, f"{final_accuracy:.2f}%", ha="right", va="bottom")

    # Combine legends
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="upper right")

    plt.tight_layout()  # Adjust spacing between subplots
    filename = f"{model_path}.png"
    plt.savefig(filename)


def plot_stats_seperate(stats: dict[str, list[float]], save_path: str, final_accuracy: float) -> None:
    """Plot the training stats."""
    batches_per_epoch = int(len(stats["train_loss"]) / len(stats["valid_loss"]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(stats["train_loss"], label="Training Loss")
    valid_loss_x = range(0, len(stats["train_loss"]), batches_per_epoch)
    plt.plot(valid_loss_x, stats["valid_loss"], label="Validation Loss", linestyle="--", marker="o")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(stats["accuracy"])), stats["accuracy"], label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("%")
    plt.title("Accuracy")
    plt.text(len(stats["accuracy"]) - 1, final_accuracy, f"{final_accuracy:.2f}%", ha="right", va="bottom")

    plt.tight_layout()  # Adjust spacing between subplots
    filename = f"{save_path}.png"
    plt.savefig(filename)
