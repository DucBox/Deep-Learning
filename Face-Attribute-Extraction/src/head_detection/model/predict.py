from ultralytics import YOLO
from paths import MODEL_DIR, TEST_IMAGE_DIR


def predict_yolo(model_path, source_dir, conf=0.25):
    """
    Perform inference with the YOLOv8 model.

    Args:
        model_path (str): Path to the trained YOLO model (e.g., best.pt).
        source_dir (str): Directory or file containing test images.
        conf (float): Confidence threshold for predictions.

    Returns:
        None
    """
    print(f"Running YOLO inference on: {source_dir}")
    model = YOLO(model_path)
    results = model.predict(source=source_dir, conf=conf, save=True)
    print(f"Inference completed. Results saved in {results}")
    return results


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
    predict_yolo(model_path=MODEL_PATH, source_dir=TEST_IMAGE_DIR, conf=0.25)
