import argparse
import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Default model paths (you can override with --keras / --tflite / --labels)
KERAS_MODEL = "best_effnetv2_adv.h5"
TFLITE_MODEL = "waste_model_fp16.tflite"
LABELS = "class_labels.json"
IMG_SIZE = (224, 224)

# Map letter codes → full waste names
CODE_TO_NAME = {
    "O": "Organic",
    "R": "Recyclable",
    "H": "Hazardous",
    "E": "E-Waste",
    "X": "Other"
}

def load_labels(labels_path):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    raw = json.load(open(labels_path, "r"))

    inv = {}

    # Case A: {"O":0,"R":1,...}
    if all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in raw.values()):
        for letter, idx in raw.items():
            idx = int(idx)
            inv[idx] = CODE_TO_NAME.get(letter, letter)
        return inv

    # Case B: {"0":"Organic", ...}
    if all(str(k).isdigit() for k in raw.keys()):
        for k, v in raw.items():
            inv[int(k)] = str(v)
        return inv

    # Fallback
    raise ValueError("Unknown label format in class_labels.json")

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_keras(image_path, model_path):
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    x = preprocess(image_path)
    preds = model.predict(x)[0]
    return int(np.argmax(preds)), float(np.max(preds))

def predict_tflite(image_path, tflite_path):
    if not os.path.exists(tflite_path):
        return None
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = preprocess(image_path)

    # quantized?
    if inp["dtype"] == np.uint8:
        scale, zp = inp["quantization"]
        if scale != 0:
            x = x / scale + zp
            x = x.astype(np.uint8)

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(out["index"])[0]

    if out["dtype"] == np.uint8:
        scale, zp = out["quantization"]
        preds = scale * (preds.astype(np.float32) - zp)

    return int(np.argmax(preds)), float(np.max(preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--keras", default=KERAS_MODEL)
    parser.add_argument("--tflite", default=TFLITE_MODEL)
    parser.add_argument("--labels", default=LABELS)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print("\n=== LABEL MAP ===")
    print(labels)

    k = predict_keras(args.image, args.keras)
    t = predict_tflite(args.image, args.tflite)

    if k:
        print(f"\nKeras → idx={k[0]} label={labels[k[0]]} conf={k[1]:.3f}")
    else:
        print("\nKeras model NOT FOUND")

    if t:
        print(f"TFLite → idx={t[0]} label={labels[t[0]]} conf={t[1]:.3f}")
    else:
        print("TFLite model NOT FOUND")
