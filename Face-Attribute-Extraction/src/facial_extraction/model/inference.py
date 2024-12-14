from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from paths import MODEL_SAVE_PATH


def predict_image(img_path, labels, threshold=0.5):
    """
    Predict the labels for a single image.

    Args:
        img_path (str): Path to the input image.
        labels (list): List of label names.
        threshold (float): Threshold for multi-label classification.

    Returns:
        list: Predicted labels.
    """
    model = load_model(MODEL_SAVE_PATH)
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    return [label for label, prob in zip(labels, predictions[0]) if prob > threshold]
