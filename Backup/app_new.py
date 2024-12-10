from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)

# Model loading
model = tf.keras.models.load_model('models/mobilenet_v2/model.h5')

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(120), nullable=False)
    upload_date = db.Column(db.DateTime, default=db.func.current_timestamp())

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    prediction_result = db.Column(db.String(120), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, default=db.func.current_timestamp())

# Routes
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Save to database
            new_image = Image(user_id=1, image_path=file_path)  # Assuming user_id=1 for simplicity
            db.session.add(new_image)
            db.session.commit()
            # Predict
            prediction_result, confidence_score = predict_image(file_path)
            new_prediction = Prediction(image_id=new_image.id, prediction_result=prediction_result, confidence_score=confidence_score)
            db.session.add(new_prediction)
            db.session.commit()
            flash('Image uploaded and prediction made successfully')
            return redirect(url_for('predictions'))
    return render_template('upload.html')

@app.route('/predictions')
def predictions():
    predictions = Prediction.query.all()
    return render_template('predictions.html', predictions=predictions)

@app.route('/delete/<int:id>', methods=['POST'])
def delete_prediction(id):
    prediction = Prediction.query.get_or_404(id)
    db.session.delete(prediction)
    db.session.commit()
    flash('Prediction deleted successfully')
    return redirect(url_for('predictions'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    return predicted_class, confidence_score

if __name__ == '__main__':
    app.run(debug=True)