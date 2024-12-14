from ultralytics import YOLO
from paths import MODEL_DIR, DATASET_DIR


def validate_yolo(model_path, data_path):
    """
    Validate the YOLOv8 model on the validation dataset.

    Args:
        model_path (str): Path to the trained YOLO model (e.g., best.pt).
        data_path (str): Path to the YOLO dataset YAML file.

    Returns:
        None
    """
    print(f"Validating YOLO model with data: {data_path}")
    model = YOLO(model_path)
    results = model.val(data=data_path)
    print("Validation completed.")
    return results


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
    DATA_PATH = os.path.join(DATASET_DIR, "data.yaml")

    validate_yolo(model_path=MODEL_PATH, data_path=DATA_PATH)
