import unittest
import numpy as np
from app import predict_image

class ModelTestCase(unittest.TestCase):
    def test_predict_image(self):
        # Simulate an image path
        image_path = 'test_image.jpg'
        prediction_result, confidence_score = predict_image(image_path)
        self.assertIsInstance(prediction_result, int)
        self.assertGreaterEqual(confidence_score, 0.0)
        self.assertLessEqual(confidence_score, 1.0)

if __name__ == '__main__':
    unittest.main()