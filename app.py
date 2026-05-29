from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from digit_recognizer import preprocess_pil_image
import numpy as np
from PIL import Image
import io
import base64
import os

# Initialize Flask app with static and template folders
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Load model
try:
    model = load_model('digit_model.h5')
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

def preprocess_image(image_bytes):
    # Use the same preprocessing used by the training/utility module to
    # ensure consistent resizing, inversion and padding (ImageOps.fit).
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # delegate to shared preprocessing which returns shape (1,28,28,1)
    arr = preprocess_pil_image(pil_img)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        img_data = data['image']
        header, encoded = img_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)

        img = preprocess_image(image_bytes)
        prediction = model.predict(img, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({'digit': digit, 'confidence': confidence})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
