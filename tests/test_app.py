import unittest
from app import app, db
from models import Image, Prediction, User

class SystemTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_image_upload_and_prediction(self):
        # Create a test user
        user = User(username='testuser', password='password', email='test@example.com')
        db.session.add(user)
        db.session.commit()

        # Simulate image upload
        with open('test_image.jpg', 'rb') as img:
            response = self.app.post('/upload', data={'image': img}, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Image uploaded successfully', response.data)

        # Check if the image is stored in the database
        image = Image.query.filter_by(user_id=user.id).first()
        self.assertIsNotNone(image)

        # Check if the prediction is stored in the database
        prediction = Prediction.query.filter_by(image_id=image.id).first()
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction.confidence_score, 0.0)
        self.assertLessEqual(prediction.confidence_score, 1.0)

if __name__ == '__main__':
    unittest.main()