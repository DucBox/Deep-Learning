import os

# Project Root Directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Paths to Data Files
def get_data_path(sub_path):
    """
    Get the full path to a data file or directory.

    Args:
        sub_path (str): The relative path from the 'data/' directory.

    Returns:
        str: The full path to the data file or directory.
    """
    return os.path.join(PROJECT_ROOT, "data", sub_path)

TRAIN_CSV = get_data_path("facial_extraction/train/_classes.csv")
VALID_CSV = get_data_path("facial_extraction/valid/_classes.csv")
TRAIN_DIR = get_data_path("facial_extraction/train/images")
VALID_DIR = get_data_path("facial_extraction/valid/images")

# Paths to Model and Logs
def get_output_path(sub_path):
    """
    Get the full path to an output file or directory.

    Args:
        sub_path (str): The relative path from the 'outputs/' directory.

    Returns:
        str: The full path to the output file or directory.
    """
    return os.path.join(PROJECT_ROOT, "outputs", sub_path)

MODEL_SAVE_PATH = get_output_path("models/best_model.h5")
LOG_PATH = get_output_path("logs/training_log.csv")

# Example usage
if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Train CSV Path:", TRAIN_CSV)
    print("Validation CSV Path:", VALID_CSV)
    print("Model Save Path:", MODEL_SAVE_PATH)
    print("Log Path:", LOG_PATH)
