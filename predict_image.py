import tensorflow as tf
import cv2
import numpy as np
import sys
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 128
MODEL_PATH = "model/deepfake_model.h5"
THRESHOLD = 0.5
# ----------------------------------------

if len(sys.argv) < 2:
    print("❌ Usage: python predict_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("❌ Image not found!")
    sys.exit(1)

# Load model
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load image
img = cv2.imread(image_path)
if img is None:
    print("❌ Invalid image")
    sys.exit(1)

# Preprocess
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img, verbose=0)[0][0]

print("\n📊 Prediction Result")
print(f"Raw Score: {prediction:.4f}")

# Correct label logic
if prediction < THRESHOLD:
    print("🟥 FAKE IMAGE")
    print(f"Confidence: {(1 - prediction) * 100:.2f}%")
else:
    print("🟩 REAL IMAGE")
    print(f"Confidence: {prediction * 100:.2f}%")
