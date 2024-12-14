from facial_extraction.paths import MODEL_DIR as FACIAL_MODEL_DIR
from head_detection.paths import MODEL_DIR as HEAD_MODEL_DIR, DATASET_DIR
from head_detection.model.train import train_yolo
from facial_extraction.model.train import build_resnet_model, train_model
from data.load_data import load_data
from utils.logger import setup_logger


def main():
    logger = setup_logger()

    logger.info("Starting the pipeline...")

    # Train Facial Extraction Model
    logger.info("Loading data for facial extraction...")
    train_generator, valid_generator = load_data()
    logger.info("Building and training the ResNet model...")
    model = build_resnet_model()
    train_model(model, train_generator, valid_generator)

    # Train YOLO Model
    logger.info("Training YOLO model...")
    train_yolo(
        model_path=HEAD_MODEL_DIR + "/last.pt",
        data_path=DATASET_DIR + "/data.yaml",
    )

    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
