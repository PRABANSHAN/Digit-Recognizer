from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

model = load_model('digit_model.h5')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img  # invert colors (assuming white digit on dark bg)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']
    header, encoded = img_data.split(',', 1)
    image_bytes = base64.b64decode(encoded)

    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))

    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)
