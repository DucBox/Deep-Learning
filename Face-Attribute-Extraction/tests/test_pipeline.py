import unittest
from end_to_end.pipeline import run_pipeline
from facial_extraction.paths import MODEL_DIR as FACIAL_MODEL_DIR
from head_detection.paths import MODEL_DIR as HEAD_MODEL_DIR, TEST_IMAGE_DIR


class TestPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        yolo_model_path = HEAD_MODEL_DIR + "/best.pt"
        resnet_model_path = FACIAL_MODEL_DIR + "/resnet_best.h5"
        test_image = TEST_IMAGE_DIR + "/test_image.jpg"

        try:
            results = run_pipeline(yolo_model_path, resnet_model_path, test_image, labels=["beard", "hat"])
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
        except Exception as e:
            self.fail(f"Pipeline execution failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
