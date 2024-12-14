import unittest
from head_detection.model.train import train_yolo
from head_detection.paths import MODEL_DIR


class TestHeadDetection(unittest.TestCase):
    def test_training(self):
        # Mock parameters
        model_path = "models/last.pt"
        data_path = DATASET_DIR + "/data.yaml"
        try:
            train_yolo(model_path, data_path, epochs=1)
            self.assertTrue(True)  # Pass if no exception
        except Exception as e:
            self.fail(f"YOLO training failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
