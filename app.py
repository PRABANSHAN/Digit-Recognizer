from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model with error handling
try:
    model = load_model('digit_model.h5')
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ Error: digit_model.h5 not found. Train the model first.")
    model = None

def preprocess_image(image_bytes):
    """Preprocess image data for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
        img = img.resize((28, 28))
        img = np.array(img)
        img = 255 - img  # invert colors (assuming white digit on dark bg)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from image"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        img_data = data['image']
        
        # Decode base64 image
        if ',' in img_data:
            header, encoded = img_data.split(',', 1)
        else:
            encoded = img_data
        
        try:
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            return jsonify({'error': 'Invalid base64 image'}), 400
        
        # Preprocess and predict
        img = preprocess_image(image_bytes)
        if img is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        prediction = model.predict(img, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        
        return jsonify({
            'digit': digit,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Run on all network interfaces (0.0.0.0) so it works on live links
    print("Starting Digit Recognizer Server...")
    print("Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
