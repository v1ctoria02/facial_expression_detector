
import logging
import os

from model import ConvolutionalNetwork
from train import evaluate, train, load_images_from_folder

LOGGER = logging.getLogger(__name__)
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    # Load images
    images_path = os.path.join(ROOT_DIR, "data")
    LOGGER.info("Loading images from %s", images_path)
    train_loader, test_loader = load_images_from_folder(images_path)
    LOGGER.info("Number of images: %d", len(train_loader))

    model = ConvolutionalNetwork()

    train(model, train_loader)
    evaluate(model, test_loader)


if __name__ == "__main__":
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

    main()
