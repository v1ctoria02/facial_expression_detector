import csv
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMAGE_SIZE, LABELS

LOGGER = logging.getLogger(__name__)


def predict_expression(frame, model: nn.Module, face_points: tuple = None) -> tuple[str, list[float]]:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if face_points is not None:
        # get tuple values of rectangle
        (x, y, w, h) = face_points
        # Crop image
        face_img = gray[y : y + h, x : x + w]

    # Transform image
    face_img = cv2.resize(face_img, IMAGE_SIZE)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=0)
    # Normalize image
    face_img = face_img / 255.0

    # To torch tensor
    face_tensor = torch.from_numpy(face_img).type(torch.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        output = model(face_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.max(probabilities, 1)[1]
        predicted_class = LABELS[predicted.item()]

    rounded_probabilities = [round(prob, 2) for prob in probabilities[0].cpu().numpy()]
    LOGGER.debug("Probabilities: %s", list(zip(LABELS, rounded_probabilities)))
    return predicted_class, rounded_probabilities


def webcam_input(model: nn.Module):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        # Capture frame-by-frame
        _ret, frame = cap.read()

        # Detect the faces using built in CV2 function
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        # Detect rectangle of each face in the frame
        for x, y, w, h in faces:
            # Get expression prediction
            predicted_expression, _ = predict_expression(frame, model, (x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow("Webcam - Facial Expression Detector", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


def validate_to_csv(images_path: str, model: nn.Module):
    # List to hold all rows for the CSV
    csv_rows = [["filepath"] + LABELS + ["predicted_label"]]
    # Process images
    for image_name in os.listdir(images_path):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Load image
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            # Predict facial expression
            predicted_class, rounded_probabilities = predict_expression(image, model)

            csv_rows.append([image_path] + rounded_probabilities + [predicted_class])

    csv_file_path = os.path.join(images_path, "classification_scores.csv")
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)

    LOGGER.info("Classification scores have been saved to %s", csv_file_path)
