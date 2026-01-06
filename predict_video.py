import tensorflow as tf
import cv2
import numpy as np
import sys
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 128
MODEL_PATH = "model/deepfake_model.h5"
FRAME_SKIP = 5
THRESHOLD = 0.5
# ----------------------------------------

if len(sys.argv) < 2:
    print("❌ Usage: python predict_video.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

if not os.path.exists(video_path):
    print("❌ Video file not found!")
    sys.exit(1)

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(video_path)

fake_frames = 0
real_frames = 0
frame_count = 0

print("🎥 Analyzing video frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)

    prediction = model.predict(frame, verbose=0)[0][0]

    if prediction < THRESHOLD:
        fake_frames += 1
    else:
        real_frames += 1

cap.release()

total = fake_frames + real_frames

print("\n📊 VIDEO ANALYSIS RESULT")
print(f"🟥 Fake Frames : {fake_frames}")
print(f"🟩 Real Frames : {real_frames}")
print(f"🎞 Total Analyzed Frames : {total}")

if total == 0:
    print("⚠️ No frames analyzed!")
    sys.exit(0)

fake_ratio = fake_frames / total
real_ratio = real_frames / total

if fake_ratio > real_ratio:
    print(f"\n🚨 FAKE VIDEO")
    print(f"Confidence: {fake_ratio * 100:.2f}%")
else:
    print(f"\n✅ REAL VIDEO")
    print(f"Confidence: {real_ratio * 100:.2f}%")
