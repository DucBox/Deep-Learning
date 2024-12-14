import os

# Dynamic Project Root
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Directories
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "head_detection")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "head_detection")
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, "test_images")

# Example Usage
if __name__ == "__main__":
    print("Dataset Directory:", DATASET_DIR)
    print("Model Directory:", MODEL_DIR)
    print("Test Image Directory:", TEST_IMAGE_DIR)
