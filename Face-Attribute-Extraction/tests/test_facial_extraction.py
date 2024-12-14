import unittest
from facial_extraction.model.train import build_resnet_model
from data.load_data import load_data


class TestFacialExtraction(unittest.TestCase):
    def test_model_building(self):
        model = build_resnet_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 175)  # ResNet50 has 175 layers

    def test_data_loading(self):
        train_generator, valid_generator = load_data()
        self.assertGreater(len(train_generator), 0)
        self.assertGreater(len(valid_generator), 0)


if __name__ == "__main__":
    unittest.main()
