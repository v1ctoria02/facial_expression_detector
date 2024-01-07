import csv
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


LOGGER = logging.getLogger(__name__)


def validate_to_csv(images_path: str, model_path: str, model: nn.Module):
    # Define transformations for the image
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((48, 48)), transforms.ToTensor()])

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading model from %s as %s", model_path, type(model))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    # List to hold all rows for the CSV
    csv_rows = [["filepath"] + LABELS + ["predicted_label"]]
    # Process images
    for image_name in os.listdir(images_path):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path).convert("L")
            image = data_transform(image)
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                probabilities = F.softmax(output, dim=1)
                predicted = torch.max(probabilities, 1)[1]
                predicted_class = LABELS[predicted.item()]

            rounded_probabilities = [round(prob, 2) for prob in probabilities[0].cpu().numpy()]
            csv_rows.append([image_path] + rounded_probabilities + [predicted_class])

    csv_file_path = os.path.join(images_path, "classification_scores.csv")
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)

    LOGGER.info("Classification scores have been saved to %s", csv_file_path)
