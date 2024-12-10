#from flask import Flask, request, jsonify
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
import sqlite3
from PIL import Image
import torch
from torchvision import models, transforms
import base64
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Configuration
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MobileNetV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
model.to(device)
model.eval()

# Preprocessing for MobileNetV2
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ImageNet class labels
imagenet_classes = {i: line.strip() for i, line in enumerate(open("imagenet_classes.txt"))}

# SQLite Database Configuration
DB_FILE = 'predictions.db'

# Database Helper Functions
def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            thumbnail TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(prediction, confidence, thumbnail):
    """Insert a prediction into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (prediction, confidence, thumbnail)
        VALUES (?, ?, ?)
    ''', (prediction, confidence, thumbnail))
    conn.commit()
    conn.close()

def get_predictions():
    """Retrieve predictions from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, prediction, confidence, thumbnail FROM predictions')
    results = cursor.fetchall()
    conn.close()
    return results

def delete_prediction(prediction_id):
    """Delete a prediction by ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    conn.commit()
    conn.close()

# Utility Functions
def allowed_file(filename):
    """Check if the file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path):
    """Classify an image using MobileNetV2."""
    image = Image.open(image_path).convert('RGB')
    input_tensor = imagenet_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, class_idx = torch.max(probabilities, 0)
    prediction = imagenet_classes[class_idx.item()]
    return prediction, confidence.item()

# Initialize database
init_db()

# API Endpoints
@app.route('/')
def index():
    return render_template('ImagePredictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    # Perform prediction
    try:
        prediction, confidence = classify_image(filepath)

        # Create thumbnail for storage
        with Image.open(filepath) as img:
            img.thumbnail((100, 100))
            thumbnail_bytes = img.tobytes()

            thumbnail_io = BytesIO()
            img.save(thumbnail_io, format="JPEG")  # Save as JPEG (you can choose another format)
            thumbnail_io.seek(0)  # Move to the beginning of the BytesIO stream

            # Encode the thumbnail to base64
            thumbnail_base64 = base64.b64encode(thumbnail_io.read()).decode('utf-8')

        # Save prediction to database
        #insert_prediction(prediction, confidence, thumbnail_bytes)
        insert_prediction(prediction, confidence, thumbnail_base64)
        print(prediction)
        print(confidence)
        response_data = {
            'prediction': prediction,
            'confidence': confidence,
        }
        return jsonify(response_data), 200
        #return jsonify({'prediction': prediction, 'confidence': confidence}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_predictions', methods=['GET'])
def get_all_predictions():
    """Retrieve all predictions."""
    results = get_predictions()
    predictions = [
        {
            'id': row[0],
            'prediction': row[1],
            'confidence': row[2],
            'thumbnail':  row[3]

        }
        for row in results
    ]
    return jsonify(predictions), 200

@app.route('/delete_prediction/<int:prediction_id>', methods=['DELETE'])
def delete_prediction_api(prediction_id):
    """Delete a specific prediction."""
    try:
        delete_prediction(prediction_id)
        return jsonify({'message': 'Prediction deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)