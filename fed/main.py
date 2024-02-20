import argparse
import logging
import os

import torch
import torch.nn as nn
from model import ExpressionNet, OldExpressionNet, XpressionNet
from train import evaluate, load_images_from_folder, train
from validation import validate_to_csv, webcam_input

import fed.config as cfg
from fed.plots import plot_stats

_logger = logging.getLogger(__name__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODELS = {
    "ExpressionNet": ExpressionNet,
    "XpressionNet": XpressionNet,
    "OldExpressionNet": OldExpressionNet,
}


def get_model_from_path(model_path: str) -> nn.Module:
    """Return the model class based on the model path."""
    filename = os.path.basename(model_path)
    model_name = filename.split("_")[0]
    model = MODELS.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found in MODELS")
    num_classes = len(cfg.LABELS)
    return model(num_classes)


def load_model_from_path(model_path: str, model: nn.Module) -> None:
    """Load the trained model and return class name of the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Loading model from %s as %s", model_path, type(model).__name__)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


def main() -> None:
    """Train the model or classify images and output scores to a CSV file."""
    # Initialize model
    model = get_model_from_path(args.model_path)

    if not args.is_validate:
        # Load images
        images_path = os.path.join(ROOT_DIR, "data")
        _logger.info("Loading images from %s", images_path)
        train_loader, test_loader = load_images_from_folder(images_path)
        _logger.info("Number of images: %d", len(train_loader))
        # Train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _logger.info("Current training device: %s", device)

        training_stats = train(model, train_loader, test_loader, epochs=cfg.EPOCHS, use_gpu=torch.cuda.is_available())
        loss, accuracy = evaluate(model, test_loader, use_gpu=torch.cuda.is_available())
        _logger.info("Validation loss: %f, accuracy: %f", loss, accuracy)

        if device == "cuda":
            # Move the model back to the CPU
            model = model.to("cpu")
        # Save the model
        _logger.info("Saving model to %s", args.model_path)
        torch.save(model.state_dict(), args.model_path)
        # Plot the training statistics
        plot_stats(training_stats, args.model_path, accuracy)
        return

    # Validate
    load_model_from_path(args.model_path, model)
    model.eval()
    if args.images_path is None:
        webcam_input(model)
    else:
        validate_to_csv(args.images_path, model)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the model or classify images and output scores to a CSV file.")
    parser.add_argument(
        "--validate",
        dest="is_validate",
        action="store_true",
        help="Perform validation on the given model using the specified image folder. "
        "If no folder is provided, webcam input will be used. If validate is not set, the model will be trained.",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Specify path to save or load the trained model. Include the model name in the filename.",
    )
    parser.add_argument(
        "--folder", dest="images_path", type=str, help="Path to the folder containing images for validation."
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    f_handler = logging.FileHandler("facial_expression_detector.log", mode="w", encoding="utf-8")
    f_handler.setLevel(log_level)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(log_level)
    logging.basicConfig(
        handlers=[f_handler, s_handler],
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
