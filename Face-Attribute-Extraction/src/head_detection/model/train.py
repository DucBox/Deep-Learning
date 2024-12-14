import os
from ultralytics import YOLO
from paths import MODEL_DIR, DATASET_DIR


def train_yolo(model_path, data_path, epochs=50, img_size=640, output_dir=MODEL_DIR):
    """
    Train the YOLOv8 model.

    Args:
        model_path (str): Path to the YOLO model (e.g., pre-trained weights).
        data_path (str): Path to the YOLO dataset YAML file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        output_dir (str): Directory to save the training results.

    Returns:
        None
    """
    print(f"Training YOLO model with data: {data_path}")
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs, imgsz=img_size, project=output_dir)
    print(f"Model training completed. Results saved in {output_dir}")


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = os.path.join(MODEL_DIR, "last.pt")
    DATA_PATH = os.path.join(DATASET_DIR, "data.yaml")

    train_yolo(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        epochs=50,
        img_size=640,
        output_dir=MODEL_DIR,
    )
