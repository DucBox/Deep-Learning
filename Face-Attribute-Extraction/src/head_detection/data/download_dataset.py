from roboflow import Roboflow
import config

def download_dataset():
    """
    Download the YOLO dataset from Roboflow.

    Returns:
        dataset (object): Roboflow dataset object.
    """
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
    project = rf.workspace(config.WORKSPACE_NAME).project(config.PROJECT_NAME)
    dataset = project.version(config.DATASET_VERSION).download("yolov8")
    print("Dataset downloaded successfully.")
    return dataset


if __name__ == "__main__":
    # Example usage
    dataset = download_dataset()
    print(f"Dataset downloaded at: {dataset.location}")
