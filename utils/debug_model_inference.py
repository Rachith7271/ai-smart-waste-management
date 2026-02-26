#!/usr/bin/env python3
"""
Debug script for waste classification models (Keras .h5 and TFLite).
Usage:
  python debug_model_inference.py --image "C:/path/to/test.jpg" --keras "utils/best_effnetv2_adv.h5" --tflite "utils/waste_model_fp16.tflite" --labels "utils/class_labels.json" --size 224

It will print raw outputs and top-5 predictions for:
 - Keras model (if provided)
 - TFLite interpreter (if provided)

It also tries both preprocess modes:
 - simple (img/255.0)
 - efficientnet_v2.preprocess_input
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf

def load_labels(labels_path):
    if not os.path.exists(labels_path):
        print("Labels file not found:", labels_path)
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # try to invert if mapping is code->index or index->label
    # If raw looks like {"O":0,"R":1} convert to {0: "O"} then user can map codes later
    try:
        # if values are ints -> code->idx
        if all(isinstance(v, int) for v in raw.values()):
            inv = {int(v): str(k) for k, v in raw.items()}
            return inv
    except Exception:
        pass
    # if keys are strings of ints -> index->label
    out = {}
    for k, v in raw.items():
        try:
            ik = int(k)
            out[ik] = str(v)
        except:
            # fallback: keep as-is, but we won't convert
            pass
    if out:
        return out
    # lastly, if raw is already idx->label as ints
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except Exception:
        return {}

def top_k_from_array(arr, k=5):
    arr = np.asarray(arr, dtype=np.float32)
    idxs = np.argsort(arr)[::-1][:k]
    return [(int(i), float(arr[int(i)])) for i in idxs]

def preprocess_image(path, size=(224,224), use_effnet=False):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32")
    if use_effnet:
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            arr = preprocess_input(arr)
        except Exception as e:
            print("Could not import EfficientNet preprocess_input:", e)
            arr = arr / 255.0
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def run_keras(keras_path, img_arr):
    print("Loading Keras model:", keras_path)
    model = tf.keras.models.load_model(keras_path)
    out = model.predict(img_arr)[0]
    return out

def run_tflite(tflite_path, img_arr):
    print("Loading TFLite model:", tflite_path)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outd = interpreter.get_output_details()[0]
    x = img_arr.copy()
    # handle quantized input
    if inp['dtype'] == np.uint8:
        scale, zp = inp['quantization']
        if scale == 0:
            raise RuntimeError("TFLite input quantization scale is zero")
        x = x / scale + zp
        x = x.astype(np.uint8)
    else:
        x = x.astype(np.float32)
    interpreter.set_tensor(inp['index'], x)
    interpreter.invoke()
    raw = interpreter.get_tensor(outd['index'])[0]
    # dequantize output if necessary
    if outd['dtype'] == np.uint8:
        scale_o, zp_o = outd['quantization']
        probs = scale_o * (raw.astype(np.float32) - zp_o)
    else:
        probs = raw.astype(np.float32)
    return probs

def print_results(title, raw_out, labels_map, topk=5):
    print("=== %s ===" % title)
    print("Raw output vector (first 10 values):", raw_out[:10].tolist() if len(raw_out)>10 else raw_out.tolist())
    # normalize to probs if not already
    arr = np.array(raw_out, dtype=np.float32)
    if np.sum(arr) <= 0.0 or np.sum(arr) > 1.1:
        ex = np.exp(arr - np.max(arr))
        probs = ex / np.sum(ex)
    else:
        probs = arr / (np.sum(arr) + 1e-12)
    top = top_k_from_array(probs, k=topk)
    print("Top-%d predictions:" % topk)
    for idx, score in top:
        label = labels_map.get(idx, f"Class_{idx}")
        print("  %d) %s — prob=%.4f (raw=%.6f)" % (idx, label, score, float(arr[idx])))
    print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to test image")
    p.add_argument("--keras", default="", help="Path to Keras model (.h5)")
    p.add_argument("--tflite", default="", help="Path to TFLite model (.tflite)")
    p.add_argument("--labels", default="", help="Path to class_labels.json")
    p.add_argument("--size", type=int, default=224, help="Image size (square) to resize for model")
    p.add_argument("--topk", type=int, default=5, help="Top-K to show")
    args = p.parse_args()

    labels_map = {}
    if args.labels:
        labels_map = load_labels(args.labels)
    else:
        print("No labels file specified; using numeric indices.")

    print("Using image:", args.image)
    print("Image size:", args.size)
    print("Labels loaded (sample):", dict(list(labels_map.items())[:10]))

    # two preprocessing modes
    for use_eff in (False, True):
        mode_name = "EffNet preprocess" if use_eff else "Simple /255"
        print("---- Preprocessing mode:", mode_name)
        arr = preprocess_image(args.image, size=(args.size, args.size), use_effnet=use_eff)

        if args.keras and os.path.exists(args.keras):
            try:
                keras_out = run_keras(args.keras, arr)
                print_results("Keras model output ("+mode_name+")", keras_out, labels_map, topk=args.topk)
            except Exception as e:
                print("Keras run error:", e)

        if args.tflite and os.path.exists(args.tflite):
            try:
                tflite_out = run_tflite(args.tflite, arr)
                print_results("TFLite model output ("+mode_name+")", tflite_out, labels_map, topk=args.topk)
            except Exception as e:
                print("TFLite run error:", e)

if __name__ == "__main__":
    main()
