import csv
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcam.methods import SmoothGradCAMpp

from fed.config import IMAGE_SIZE, LABELS, RECTANGLE_COLOR, TEXT_COLOR

_logger = logging.getLogger(__name__)


def _predict_expression(
    frame: np.ndarray, model: nn.Module, use_gradcam: bool = True, rectangle: tuple | None = None
) -> tuple[str, list[float]]:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if rectangle is not None:
        # Unpack rectangle
        (x, y, w, h) = rectangle
        # Crop image
        face_img = gray[y : y + h, x : x + w]

    # Transform image
    face_img = cv2.resize(face_img, IMAGE_SIZE)
    face_img = np.expand_dims(face_img, axis=(0, 1))
    # Normalize image
    face_img = face_img / 255.0

    # To torch tensor
    face_tensor = torch.from_numpy(face_img).type(torch.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_tensor = face_tensor.to(device)
    if use_gradcam:
        # with SmoothGradCAMpp(model, model.conv1, input_shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1])) as cam_extractor:
        with SmoothGradCAMpp(model, model.conv4, input_shape=(512, 6, 6)) as cam_extractor:
            out = model(face_tensor)
            probabilities = F.softmax(out, dim=1)
            predicted = torch.max(probabilities, 1)[1]
            predicted_class = LABELS[predicted.item()]
            # Get the CAM which will map the input image to the output class
            activation_map = cam_extractor(predicted.item(), out)
            # Merge frame's rectangle with gradcam activation map
            heatmap = cv2.resize(activation_map[0].squeeze(0).cpu().numpy(), (w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_VIRIDIS)
            frame[y : y + h, x : x + w] = cv2.addWeighted(frame[y : y + h, x : x + w], 1, heatmap, 0.5, gamma=0)

    else:
        with torch.no_grad():
            out = model(face_tensor)
            probabilities = F.softmax(out, dim=1)
            predicted = torch.max(probabilities, 1)[1]
            predicted_class = LABELS[predicted.item()]

    rounded_probabilities = [round(prob, 2) for prob in probabilities[0].cpu().detach().numpy()]
    _logger.debug("Probabilities: %s", list(zip(LABELS, rounded_probabilities, strict=False)))

    return predicted_class, rounded_probabilities


def webcam_input(model: nn.Module) -> None:
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise OSError("Cannot open webcam")

    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if length > 0:
        _logger.info("Video length: %d", length)
    _logger.info("Video info: size: %dx%d, FPS: %d", width, height, fps)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        # Capture frame-by-frame
        _ret, frame = cap.read()
        if not _ret:
            _logger.error("Error reading frame")
            break

        # Detect the faces using built-in CV2 function
        face_rois = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=4, minSize=(36, 36))

        # Detect rectangle of each face in the frame
        for rectangle in face_rois:
            # Get expression prediction
            predicted_expression, _ = _predict_expression(frame, model, rectangle=rectangle)
            # unpack rectangle
            (x, y, w, h) = rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
            cv2.putText(
                frame, f"Emotion: {predicted_expression}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2
            )

        # Metadata on the frame
        cv2.putText(
            frame, f"Number of faces detected: {len(face_rois)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1
        )
        # Display the frame
        cv2.imshow("Webcam - Facial Expression Detector", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


def validate_to_csv(images_path: str, model: nn.Module) -> None:
    # List to hold all rows for the CSV
    csv_rows = ["filepath", *LABELS, "predicted_label"]
    # Process images
    for image_name in os.listdir(images_path):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Load image
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            # Predict facial expression
            predicted_class, rounded_probabilities = _predict_expression(image, model, use_gradcam=False)

            csv_rows.append([image_path, *rounded_probabilities, predicted_class])

    csv_file_path = os.path.join(images_path, "classification_scores.csv")
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)

    _logger.info("Classification scores have been saved to %s", csv_file_path)
