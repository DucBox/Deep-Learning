import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.logger import log 
from facial_extraction.config import LABELS  


def preprocess_image(image, size=(224, 224)):
    """
    Preprocess an image for ResNet50.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def crop_and_classify(image, bbox, resnet_model, labels, threshold=0.5):
    """
    Crop an image based on the bounding box and classify using ResNet50.
    """
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]
    input_data = preprocess_image(cropped_image)
    predictions = resnet_model.predict(input_data)

    classified_labels = {
        label: prob for label, prob in zip(labels, predictions[0]) if prob > threshold
    }
    return classified_labels


def run_pipeline(yolo_model_path, resnet_model_path, image_path, labels):
    """
    Run the end-to-end pipeline: YOLO detection + ResNet classification.
    """
    # Load models
    yolo_model = YOLO(yolo_model_path)
    resnet_model = load_model(resnet_model_path)

    # Read and process the image
    image = cv2.imread(image_path)

    # Log the image path
    log(f"Processing image: {image_path}")

    # Run YOLO inference
    results = yolo_model(image_path)

    # Process each detected bounding box
    final_results = []
    for r in results:
        for bbox_tensor in r.boxes.xyxy:
            bbox = bbox_tensor.numpy().astype(int)
            classifications = crop_and_classify(image, bbox, resnet_model, labels)
            final_results.append({"bbox": bbox, "classifications": classifications})

    return final_results
