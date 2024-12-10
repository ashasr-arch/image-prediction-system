import unittest
from app import app, db
from models import User, Image, Prediction

class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_user_creation(self):
        user = User(username='testuser', password='password', email='test@example.com')
        db.session.add(user)
        db.session.commit()
        self.assertIsNotNone(User.query.filter_by(username='testuser').first())

    def test_image_creation(self):
        user = User(username='testuser', password='password', email='test@example.com')
        db.session.add(user)
        db.session.commit()
        image = Image(user_id=user.id, image_path='path/to/image.jpg')
        db.session.add(image)
        db.session.commit()
        self.assertIsNotNone(Image.query.filter_by(user_id=user.id).first())

    def test_prediction_creation(self):
        user = User(username='testuser', password='password', email='test@example.com')
        db.session.add(user)
        db.session.commit()
        image = Image(user_id=user.id, image_path='path/to/image.jpg')
        db.session.add(image)
        db.session.commit()
        prediction = Prediction(image_id=image.id, prediction_result='Cat', confidence_score=0.95)
        db.session.add(prediction)
        db.session.commit()
        self.assertIsNotNone(Prediction.query.filter_by(image_id=image.id).first())

if __name__ == '__main__':
    unittest.main()