"""
digit_recognizer.py

Contains:
- build_model(): returns compiled CNN model
- train_and_save(model_path, epochs, batch_size): trains on MNIST and saves .h5
- load_model_from_file(model_path): loads and returns Keras model
- preprocess_pil_image(pil_img): convert PIL image -> model-ready numpy array
- predict_from_pil(pil_img, model): returns predicted digit and probabilities

Usage examples:
1) Train and save model:
   python digit_recognizer.py --train --model-path digit_recognizer_cnn.h5

2) Predict from image file:
   python digit_recognizer.py --predict-from-file sample_digit.png --model-path digit_recognizer_cnn.h5
"""

import argparse
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
import os
import sys
import base64
import io
import tensorflow as tf

def build_model(input_shape=(28,28,1), num_classes=10):
    """Builds and returns a compiled CNN model."""
    model = Sequential([
        # First block
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second block
        Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third block
        Conv2D(128, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten & Dense layers
        Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save(model_path='digit_recognizer_cnn.h5', epochs=30, batch_size=128, verbose=1):
    """Train on MNIST with data augmentation and save the trained model to model_path."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and normalize
    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
    x_test  = x_test.reshape(-1,28,28,1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    # Data augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    model = build_model()
    print("Training model with data augmentation...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
        steps_per_epoch=len(x_train) // batch_size
    )

    print("Evaluating on test set...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test accuracy: {acc*100:.2f}%  (loss: {loss:.4f})")

    print(f"Saving model to {model_path} ...")
    model.save(model_path)
    print("Model saved.")

def load_model_from_file(model_path='digit_recognizer_cnn.h5'):
    """Load and return a Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    return model

def preprocess_pil_image(pil_img):
    """
    Preprocess a PIL Image to the model input:
    - convert to grayscale
    - resize to 28x28
    - invert (so dark background -> 0 and white strokes -> 1 like MNIST)
    - normalize to [0,1]
    - reshape to (1,28,28,1)
    """
    # convert to grayscale
    img = pil_img.convert('L')
    # ensure square and resize: keep aspect ratio by padding
    img = ImageOps.fit(img, (28,28), Image.ANTIALIAS)

    # Decide whether to invert based on average brightness.
    # If the image is bright (light background, dark strokes), mean will be high -> invert.
    arr_uint8 = np.array(img)
    mean_val = arr_uint8.mean()
    if mean_val > 127:
        img = ImageOps.invert(img)

    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr

def predict_from_pil(pil_img, model):
    """Return (predicted_digit, probabilities_array) for a PIL image."""
    x = preprocess_pil_image(pil_img)
    probs = model.predict(x)
    pred = int(np.argmax(probs, axis=1)[0])
    return pred, probs[0]

def predict_from_base64(data_url, model):
    """
    Accepts a data URL (e.g. "data:image/png;base64,...."), decodes to image, predicts.
    Returns (predicted_digit, probabilities_array)
    """
    if ',' in data_url:
        header, encoded = data_url.split(',', 1)
    else:
        encoded = data_url
    image_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(image_bytes))
    return predict_from_pil(pil_img, model)

# CLI: allow training or single-file prediction
def main():
    parser = argparse.ArgumentParser(description="Digit recognizer train/predict utility.")
    parser.add_argument('--train', action='store_true', help='Train the model on MNIST and save to --model-path')
    parser.add_argument('--model-path', type=str, default='digit_recognizer_cnn.h5', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--predict-from-file', type=str, help='Path to image file for prediction (PNG/JPG)')
    args = parser.parse_args()

    if args.train:
        train_and_save(model_path=args.model_path, epochs=args.epochs, batch_size=args.batch_size)
        return

    if args.predict_from_file:
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}. Train first with --train")
            sys.exit(1)
        model = load_model_from_file(args.model_path)
        # open image
        pil_img = Image.open(args.predict_from_file)
        pred, probs = predict_from_pil(pil_img, model)
        print(f"Predicted digit: {pred}")
        print("Probabilities:", np.round(probs, 4))
        return

    parser.print_help()

if __name__ == '__main__':
    main()
