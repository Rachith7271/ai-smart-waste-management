# test_model.py
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

MODEL_PATH = "waste_classifier_efficientnetv2b0.h5"
LABEL_PATH = "class_labels.json"
IMG_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

with open(LABEL_PATH, "r") as f:
    index_to_class = json.load(f)

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image.")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    arr = np.expand_dims(img, axis=0).astype('float32')
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    class_name = index_to_class.get(str(idx), "Unknown")
    print(f"Prediction: {class_name} ({conf:.3f})")
    return class_name, conf

if __name__ == "__main__":
    test_image = r"C:\Users\rachi\WasteProject\dataset\TEST\O\O_12583.jpg"
    predict_image(test_image)
