# realtime_cam.py
"""
Realtime webcam classifier using the EfficientNetV2B0 model.
Press 'q' to quit.
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import time
import json
import os

MODEL_PATH = "waste_classifier_efficientnetv2b0.h5"
LABEL_PATH = "class_labels.json"
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.35   # tune to your needs
SMOOTHING = 5           # frames to average for smoothing

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train or copy it before running realtime_cam.py")
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'r') as f:
    index_to_class = json.load(f)

# ensure keys are strings for lookup
index_to_class = {str(k): v for k, v in index_to_class.items()}

def predict_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    arr = np.expand_dims(img, axis=0).astype('float32')
    arr = preprocess_input(arr)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    name = index_to_class.get(str(idx), 'Unknown')
    return name, conf

pred_buffer = []
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing the device index (0,1,2...).")

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, conf = predict_frame(frame)
    pred_buffer.append((name, conf))
    if len(pred_buffer) > SMOOTHING:
        pred_buffer.pop(0)

    # smoothing: pick the most frequent name in buffer, average its conf
    counts = {}
    confs = {}
    for n, c in pred_buffer:
        counts[n] = counts.get(n, 0) + 1
        confs[n] = confs.get(n, 0.0) + c
    best_name = max(counts.items(), key=lambda x: x[1])[0]
    avg_conf = confs[best_name] / counts[best_name]

    label_text = f"{best_name}: {avg_conf:.2f}"
    if avg_conf < CONF_THRESHOLD:
        label_text = f"Unknown ({avg_conf:.2f})"

    # FPS
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if cur_time != prev_time else 0.0
    prev_time = cur_time

    # draw overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 70), (0,0,0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow('Waste Classifier (q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
