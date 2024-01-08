import argparse
import logging
import os

import torch

import config as cfg
from model import BestNet
from train import evaluate, load_images_from_folder, train
from validation import validate_to_csv, webcam_input

LOGGER = logging.getLogger(__name__)
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_model_from_path(model_path: str, model):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading model from %s as %s", model_path, type(model))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


def main(args: argparse.Namespace):
    # Load images
    images_path = os.path.join(ROOT_DIR, "data")
    LOGGER.info("Loading images from %s", images_path)
    train_loader, test_loader = load_images_from_folder(images_path, batch_size=32)
    LOGGER.info("Number of images: %d", len(train_loader))

    num_classes = len(cfg.LABELS)
    # model = ResNet(num_classes)
    model = BestNet(num_classes)

    if not args.is_validate:
        # train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        LOGGER.info("Current training device: %s", device)

        train(model, train_loader, epochs=cfg.EPOCHS, use_gpu=torch.cuda.is_available())
        evaluate(model, test_loader, use_gpu=torch.cuda.is_available())

        if device == "cuda":
            model = model.to("cpu")
        LOGGER.info("Saving model to %s", args.model_path)
        torch.save(model.state_dict(), args.model_path)
    else:
        load_model_from_path(args.model_path, model)
        if args.images_path is None:
            webcam_input(model)
        else:
            validate_to_csv(args.images_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model or classify images and output scores to a CSV file.")
    parser.add_argument(
        "--validate", dest="is_validate", action="store_true", help="Validate the given model on the given image folder"
    )
    parser.add_argument(
        "--model", dest="model_path", type=str, required=True, help="Path to save or load the model file"
    )
    parser.add_argument("--folder", dest="images_path", type=str, help="Path to the folder containing images")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    f_handler = logging.FileHandler("facial_expression_detector.log", mode="w")
    f_handler.setLevel(log_level)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(log_level)
    logging.basicConfig(
        handlers=[f_handler, s_handler],
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main(args)
