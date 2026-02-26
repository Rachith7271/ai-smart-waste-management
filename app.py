# app.py  (updated by assistant) - paste entire file replacing your old app.py
import os
import csv
import json
import threading
import time
import datetime
import numpy as np
import pandas as pd
import joblib
import requests
import base64
from PIL import Image
from io import StringIO, BytesIO

from flask import Flask, render_template, request, jsonify, url_for, session, redirect, flash, current_app
from flask_socketio import SocketIO, emit, join_room, leave_room

# ML imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression

# For optional EfficientNet preprocessing (toggle below)
try:
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
except Exception:
    effnet_preprocess = None

# ----------------------------
# Configuration / Environment
# ----------------------------
SIMULATION = os.getenv("SIMULATION", "1") == "1"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Sensor CSV files
HIST_SENSOR_CSV = os.path.join(DATA_DIR, "sensor_full_5years_hourly.csv")
LIVE_SENSOR_CSV = os.path.join(DATA_DIR, "sensor_live_append.csv")
SENSOR_AGG_CSV = os.path.join(DATA_DIR, "sensor_readings.csv")

# Forecast dataset (existing names you used)
FORECAST_CSV_PRIMARY = os.path.join(DATA_DIR, "waste_data_extended_2017_2024.csv")
FORECAST_CSV_ALT = os.path.join(DATA_DIR, "cleaned_waste_data.csv")

# Models path
MODEL_CLASS_PATH = os.path.join("models", "waste_classifier.h5")
MODEL_FORECAST_PATH = os.path.join("models", "waste_forecast_model.pkl")
AREA_ENCODER_PATH = os.path.join("models", "area_encoder.pkl")
CLASS_LABELS_JSON = os.path.join("models", "class_labels.json")

# Also try utils path for labels if present
UTILS_LABELS_JSON = os.path.join("utils", "class_labels.json")

# TFLite candidates (check these paths at startup)
TFLITE_CANDIDATES = [
    os.path.join("utils", "waste_model_fp16.tflite"),
    os.path.join("models", "waste_model_fp16.tflite"),
    os.path.join("utils", "waste_model.tflite"),
    os.path.join("models", "waste_model.tflite")
]

# ORS Key
OPENROUTESERVICE_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjVhMTFkNTk3YmM2YzRlYzM4OWRiNmNjNTIwZTYwN2Q5IiwiaCI6Im11cm11cjY0In0="
ORS_OPT_KEY = OPENROUTESERVICE_API_KEY

# ----------------------------
# App & SocketIO
# ----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# ----------------------------
# MODEL / PREPROCESS SETTINGS (tweak if needed)
# ----------------------------
# The debug run used size 224 and simple /255 preprocessing that produced correct output.
IMG_SIZE_MODEL = (224, 224)         # set to the size you trained with (e.g., (128,128) or (224,224))
USE_EFFICIENTNET_PREPROCESS = False  # set True if your training used efficientnet_v2.preprocess_input

# ----------------------------
# In-memory caches & helpers (unchanged from your app)
# ----------------------------
SENSOR_CACHE = {"df": None, "summary": [], "last_load": None}
WASTE_CACHE = {"waste_data": [], "recycle_data": {}, "trend_data": [], "top_areas": [], "last_load": None}

def fast_read_sensor_csv(path, nrows=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    if nrows is None:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            return df
        except Exception as e:
            print(f"[fast_read_sensor_csv] full read failed {path}: {e}")
            return pd.DataFrame()
    else:
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                data = b''
                lines = 0
                blocksize = 4096
                while lines < nrows + 5 and f.tell() > 0:
                    step = min(blocksize, f.tell())
                    f.seek(-step, os.SEEK_CUR)
                    chunk = f.read(step)
                    f.seek(-step, os.SEEK_CUR)
                    data = chunk + data
                    lines = data.count(b'\n')
                    if f.tell() == 0:
                        break
                text = data.decode(errors='ignore')
                last_lines = text.splitlines()[-nrows:]
                if not last_lines:
                    return pd.DataFrame()
                csv_text = "\n".join(last_lines)
                df = pd.read_csv(StringIO(csv_text), dtype=str, low_memory=False)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                return df
        except Exception as e:
            try:
                df = pd.read_csv(path, dtype=str, low_memory=False)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                return df
            except Exception as e2:
                print("[fast_read_sensor_csv] fallback full read failed:", e2)
                return pd.DataFrame()

def load_sensor_data(force_reload=False, max_rows_from_history=50000):
    now = pd.Timestamp.utcnow()
    cache_ttl = pd.Timedelta(seconds=5)
    last = SENSOR_CACHE.get("last_load")
    if not force_reload and last is not None and (now - last) < cache_ttl and SENSOR_CACHE.get("df") is not None:
        return SENSOR_CACHE["df"]

    frames = []
    if os.path.exists(LIVE_SENSOR_CSV):
        try:
            df_live = pd.read_csv(LIVE_SENSOR_CSV, dtype=str, low_memory=False)
            df_live.columns = [c.strip().lower().replace(" ", "_") for c in df_live.columns]
            frames.append(df_live)
        except Exception as e:
            print("[load_sensor_data] cannot read LIVE_SENSOR_CSV:", e)

    if os.path.exists(HIST_SENSOR_CSV):
        try:
            df_hist = fast_read_sensor_csv(HIST_SENSOR_CSV, nrows=max_rows_from_history)
            if not df_hist.empty:
                frames.append(df_hist)
        except Exception as e:
            print("[load_sensor_data] cannot read HIST_SENSOR_CSV:", e)

    if not frames:
        df = pd.DataFrame(columns=["timestamp","node_id","area","lat","lon","fill_level_pct","temperature_c","battery_v","sensor_status"])
        SENSOR_CACHE["df"] = df
        SENSOR_CACHE["last_load"] = pd.Timestamp.utcnow()
        return df

    df = pd.concat(frames, ignore_index=True, sort=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp"] = pd.NaT

    if "node_id" not in df.columns and "device_id" in df.columns:
        df["node_id"] = df["device_id"]
    if "node_id" not in df.columns:
        df["node_id"] = df.index.map(lambda i: f"node_{i}")

    if "latitude" in df.columns and "lat" not in df.columns:
        df["lat"] = df["latitude"]
    if "longitude" in df.columns and "lon" not in df.columns:
        df["lon"] = df["longitude"]
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

    for key in ("fill_level_pct", "fill_level_percent", "level_percent", "fill"):
        if key in df.columns:
            df[key] = df[key].astype(str).str.replace('%','', regex=False)

    def first_numeric_fill(row):
        for k in ("fill_level_pct","fill_level_percent","level_percent","fill"):
            if k in row and pd.notna(row[k]) and str(row[k]).strip() != '':
                try:
                    return float(row[k])
                except:
                    continue
        return 0.0
    df["fill_level_pct"] = df.apply(first_numeric_fill, axis=1)

    if "temperature_c" in df.columns:
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce").fillna(0)
    else:
        df["temperature_c"] = 0
    if "battery_v" in df.columns:
        df["battery_v"] = pd.to_numeric(df["battery_v"], errors="coerce").fillna(0)
    else:
        df["battery_v"] = 0

    final_cols = ["timestamp","node_id","area","lat","lon","fill_level_pct","temperature_c","battery_v","sensor_status"]
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA

    SENSOR_CACHE["df"] = df[final_cols]
    SENSOR_CACHE["last_load"] = pd.Timestamp.utcnow()
    return SENSOR_CACHE["df"]

def get_current_sensor_summary(df):
    if df is None or df.empty:
        return []

    if "timestamp" in df.columns:
        df_sorted = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        df_sorted = df

    try:
        last = df_sorted.groupby("node_id").last().reset_index()
    except Exception:
        last = df_sorted.drop_duplicates(subset=["node_id"], keep="last").copy()

    out = []
    for _, r in last.iterrows():
        out.append({
            "node_id": str(r.get("node_id")),
            "area": str(r.get("area")) if pd.notna(r.get("area")) else "",
            "lat": float(r.get("lat")) if pd.notna(r.get("lat")) else None,
            "lon": float(r.get("lon")) if pd.notna(r.get("lon")) else None,
            "fill_level_pct": float(r.get("fill_level_pct") or 0),
            "temperature_c": float(r.get("temperature_c") or 0),
            "battery_v": float(r.get("battery_v") or 0),
            "last_seen": pd.to_datetime(r.get("timestamp")).isoformat() if pd.notna(r.get("timestamp")) else None,
            "sensor_status": r.get("sensor_status") if pd.notna(r.get("sensor_status")) else "OK"
        })
    return out

# ----------------------------
# Label loading & mapping
# ----------------------------
def load_labels_map(path):
    """
    Returns dict: {idx (int): human_label (str)}
    Handles formats:
      {"O": 0, "R": 1}         -> code->idx (we map codes to human labels via CODE_TO_NAME)
      {"0": "Organic", ...}    -> string-index -> label
      {0: "Organic", ...}      -> int-index -> label
      ["Organic", ...]         -> list
    """
    CODE_TO_NAME = {"O":"Organic","R":"Recyclable","H":"Hazardous","E":"E-Waste","X":"Other"}

    if not os.path.exists(path):
        print("[labels] file missing:", path)
        # fallback defaults
        return {0:"Organic",1:"Recyclable",2:"E-Waste",3:"Hazardous",4:"Other"}

    try:
        raw = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        print("[labels] json load failed:", e)
        return {0:"Organic",1:"Recyclable",2:"E-Waste",3:"Hazardous",4:"Other"}

    labels_map = {}

    # case 1: values are ints -> likely code->idx (e.g. {"O":0})
    if isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values()):
        for code, idx in raw.items():
            try:
                labels_map[int(idx)] = CODE_TO_NAME.get(str(code), str(code))
            except:
                labels_map[int(idx)] = str(code)
        return labels_map

    # case 2: keys look like integers (strings) -> {"0":"Organic"}
    try:
        possible = {}
        for k, v in raw.items():
            try:
                ik = int(k)
                possible[ik] = str(v)
            except:
                possible = {}
                break
        if possible:
            return possible
    except Exception:
        pass

    # case 3: keys are ints already
    try:
        for k, v in raw.items():
            if isinstance(k, int):
                labels_map[int(k)] = str(v)
        if labels_map:
            return labels_map
    except Exception:
        pass

    # case 4: list
    try:
        if isinstance(raw, list):
            for i, v in enumerate(raw):
                labels_map[i] = str(v)
            return labels_map
    except:
        pass

    # final fallback: enumerate
    try:
        i = 0
        for k, v in raw.items():
            labels_map[i] = str(v)
            i += 1
        return labels_map
    except:
        return {0:"Organic",1:"Recyclable",2:"E-Waste",3:"Hazardous",4:"Other"}

# pick a labels JSON (utils preferred if exists)
_labels_file = UTILS_LABELS_JSON if os.path.exists(UTILS_LABELS_JSON) else (CLASS_LABELS_JSON if os.path.exists(CLASS_LABELS_JSON) else None)
if _labels_file:
    LABELS_MAP = load_labels_map(_labels_file)
else:
    LABELS_MAP = {0:"Organic",1:"Recyclable",2:"E-Waste",3:"Hazardous",4:"Other"}

print("[labels] LABELS_MAP:", LABELS_MAP)

# ----------------------------
# Load ML models safely (if present)
# ----------------------------
classification_model = None
area_encoder = None
forecast_model = None
try:
    if os.path.exists(MODEL_CLASS_PATH):
        try:
            classification_model = load_model(MODEL_CLASS_PATH)
            print("Loaded Keras classification model:", MODEL_CLASS_PATH)
        except Exception as e:
            print("Keras model load failed:", e)
    if os.path.exists(AREA_ENCODER_PATH):
        area_encoder = joblib.load(AREA_ENCODER_PATH)
    if os.path.exists(MODEL_FORECAST_PATH):
        forecast_model = joblib.load(MODEL_FORECAST_PATH)
    print("✅ Models/labels loaded where available.")
except Exception as e:
    print("⚠ Model loading issue:", e)
    # -----------------------------------------------------
# Detect Keras model expected input size (H, W)
# -----------------------------------------------------
KERAS_EXPECTED_INPUT = None
try:
    if classification_model is not None:
        shape = getattr(classification_model, "input_shape", None)
        if shape and len(shape) >= 3:
            h = shape[1] if shape[1] else None
            w = shape[2] if shape[2] else None
            if h and w:
                KERAS_EXPECTED_INPUT = (int(h), int(w))
                print(f"[KERAS] Expected input size detected: {KERAS_EXPECTED_INPUT}")
except Exception:
    KERAS_EXPECTED_INPUT = None


# Try to find a TFLite model and initialize interpreter
TFLITE_INTERPRETER = None
TFLITE_INPUT = {}
TFLITE_OUTPUT = {}
_tflite_lock = threading.Lock()

def try_load_tflite():
    global TFLITE_INTERPRETER, TFLITE_INPUT, TFLITE_OUTPUT
    for p in TFLITE_CANDIDATES:
        if os.path.exists(p):
            try:
                interpreter = tf.lite.Interpreter(model_path=p)
                interpreter.allocate_tensors()
                inp = interpreter.get_input_details()[0]
                outd = interpreter.get_output_details()[0]
                TFLITE_INTERPRETER = interpreter
                TFLITE_INPUT = {"index": inp["index"], "shape": tuple(inp["shape"]), "dtype": inp["dtype"], "quantization": inp.get("quantization", (0.0, 0))}
                TFLITE_OUTPUT = {"index": outd["index"], "shape": tuple(outd["shape"]), "dtype": outd["dtype"], "quantization": outd.get("quantization", (0.0, 0))}
                print("[tflite] Loaded interpreter:", p, "input:", TFLITE_INPUT, "output:", TFLITE_OUTPUT)
                return
            except Exception as e:
                print("[tflite] failed to load", p, e)
    print("[tflite] no valid tflite model found in candidates.")

try_load_tflite()

# ----------------------------
# Disposal info & classes (unchanged)
# ----------------------------
classes = ["Organic", "Recyclable", "Hazardous", "E-Waste", "Other"]

disposal_info = {
    "Organic": {"method": "Compost/green bin.", "videos": ["https://www.youtube.com/embed/zy70DAaeFBI"]},
    "Recyclable": {"method": "Rinse & sort.", "videos": ["https://www.youtube.com/embed/65KuQhwk92g"]},
    "Hazardous": {"method": "Take to hazardous center.", "videos": ["https://www.youtube.com/embed/7h4vS9Zd2g0"]},
    "E-Waste": {"method": "Recycle at e-waste centers.", "videos": ["https://www.youtube.com/embed/-M9RBW3bsCQ"]},
    "Other": {"method": "Follow local guidelines.", "videos": ["https://www.youtube.com/embed/VvqQ1wRXcP0"]}
}

# ----------------------------
# Preprocessing helpers
# ----------------------------
def _pil_load_and_resize_from_bytes(img_bytes, size=IMG_SIZE_MODEL):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        return img
    except Exception as e:
        raise RuntimeError(f"failed to read/resize image bytes: {e}")

def _preprocess_from_bytes(img_bytes):
    """
    Returns a numpy array shaped (1,H,W,3), dtype float32, preprocessed according to config.
    """
    img = _pil_load_and_resize_from_bytes(img_bytes, size=IMG_SIZE_MODEL)
    arr = np.asarray(img).astype(np.float32)
    if USE_EFFICIENTNET_PREPROCESS and effnet_preprocess is not None:
        arr = effnet_preprocess(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

def _preprocess_from_path(img_path):
    with open(img_path, "rb") as f:
        b = f.read()
    return _preprocess_from_bytes(b)

# ----------------------------
# Prediction helpers (TFLite + Keras)
# ----------------------------
def _to_probs_from_raw(raw_arr):
    arr = np.array(raw_arr, dtype=np.float32)
    s = np.sum(arr)
    if s <= 0.0 or s > 1.1:
        ex = np.exp(arr - np.max(arr))
        probs = ex / np.sum(ex)
    else:
        probs = arr / (s + 1e-12)
    return probs.astype(np.float32)

def _predict_tflite_array(arr):
    """
    arr: numpy array (1,H,W,3) float32 normalized as training expects.
    Returns (probs numpy array, raw_output numpy array)
    """
    global TFLITE_INTERPRETER, TFLITE_INPUT, TFLITE_OUTPUT, _tflite_lock
    if TFLITE_INTERPRETER is None:
        raise RuntimeError("TFLite interpreter not loaded")

    with _tflite_lock:
        inp_meta = TFLITE_INPUT
        out_meta = TFLITE_OUTPUT
        # prepare input according to dtype/quantization
        x = arr.copy()
        if inp_meta['dtype'] == np.uint8:
            scale, zp = inp_meta['quantization']
            if scale is None or scale == 0:
                raise RuntimeError("tflite input quantization scale invalid")
            x_in = (x / scale + zp).astype(np.uint8)
        else:
            x_in = x.astype(np.float32)

        TFLITE_INTERPRETER.set_tensor(inp_meta['index'], x_in)
        TFLITE_INTERPRETER.invoke()
        raw_out = TFLITE_INTERPRETER.get_tensor(out_meta['index'])[0]
        if out_meta['dtype'] == np.uint8:
            scale_o, zp_o = out_meta['quantization']
            if scale_o is None:
                scale_o = 1.0
            probs = scale_o * (raw_out.astype(np.float32) - zp_o)
        else:
            probs = raw_out.astype(np.float32)
    # convert to normalized probabilities
    probs_norm = _to_probs_from_raw(probs)
    return probs_norm, raw_out

KERAS_FALLBACK = classification_model  # may be None

def _predict_keras_array(arr):
    """
    Resize arr to KERAS_EXPECTED_INPUT if necessary, then predict.
    """
    if KERAS_FALLBACK is None:
        raise RuntimeError("Keras model not loaded")

    try:
        arr_for_model = arr

        # Resize ONLY if model input size differs
        if KERAS_EXPECTED_INPUT:
            expected_h, expected_w = KERAS_EXPECTED_INPUT

            if arr.shape[1] != expected_h or arr.shape[2] != expected_w:
                try:
                    arr_tf = tf.image.resize(arr, size=(expected_h, expected_w))
                    arr_for_model = arr_tf.numpy().astype(np.float32)
                    print(f"[KERAS] Resized input from {(arr.shape[1], arr.shape[2])} to {(expected_h, expected_w)}")
                except Exception as e:
                    print("Resize failed, falling back to original array:", e)
                    arr_for_model = arr

        preds_raw = KERAS_FALLBACK.predict(arr_for_model)
        preds = np.asarray(preds_raw)[0]
        probs = _to_probs_from_raw(preds)

        return probs, preds

    except Exception as e:
        print("[KERAS] Prediction failure:", e)
        raise

def predict_with_ensemble(arr):
    """
    Attempts to get probs from TFLite and/or Keras and average them.
    Returns averaged probs or None if neither available.
    """
    probs_list = []
    try:
        if TFLITE_INTERPRETER is not None:
            p_tflite, _ = _predict_tflite_array(arr)
            probs_list.append(p_tflite)
    except Exception as e:
        current_app.logger.exception("tflite predict failed")

    try:
        if KERAS_FALLBACK is not None:
            p_keras, _ = _predict_keras_array(arr)
            probs_list.append(p_keras)
    except Exception as e:
        current_app.logger.exception("keras predict failed")

    if not probs_list:
        return None
    avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    avg /= (np.sum(avg) + 1e-12)
    return avg

def classify_array(arr, prefer_tflite=True, return_top_k=3):
    """
    arr: (1,H,W,3) float32 preprocessed
    Returns (label_str, conf, idx, topk_list)
    topk_list: [{"label":..., "confidence":..., "idx":...}, ...]
    """
    # First try ensemble if both present (more robust)
    probs = predict_with_ensemble(arr)
    if probs is None:
        # fallback to single model preference
        if prefer_tflite and TFLITE_INTERPRETER is not None:
            probs, _ = _predict_tflite_array(arr)
        elif KERAS_FALLBACK is not None:
            probs, _ = _predict_keras_array(arr)
        else:
            raise RuntimeError("No model available for prediction")

    # get top-k
    idxs = np.argsort(probs)[::-1][:return_top_k]
    topk = []
    for i in idxs:
        lbl = LABELS_MAP.get(int(i), f"Class_{i}")
        topk.append({"label": lbl, "confidence": float(round(float(probs[int(i)]), 6)), "idx": int(i)})
    top1 = topk[0]
    return top1["label"], float(top1["confidence"]), int(top1["idx"]), topk

# ----------------------------
# Legacy predict_image (kept for compatibility)
# ----------------------------
def predict_image(image_path):
    global classification_model, LABELS_MAP
    if classification_model is None:
        print("⚠ classification_model not loaded")
        return None
    if not os.path.exists(image_path):
        print("❌ image not found:", image_path)
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE_MODEL)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = classification_model.predict(arr)
        idx = int(np.argmax(preds[0]))
        label = LABELS_MAP.get(idx, classes[idx] if idx < len(classes) else "Other")
        confidence = float(np.max(preds[0]))
        print(f"✅ Predicted: {label} (Confidence: {confidence:.2f})")
        return label
    except Exception as e:
        print("predict_image error:", e)
        return None

# ----------------------------
# Routes: home, classify, forecast (classify updated)
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["GET", "POST"])
def classify():
    """
    Unified classify route:
      - GET: render classify.html
      - POST:
          * multipart 'frame' -> webcam frame -> returns JSON {label, confidence, idx, top3}
          * JSON {frame: "data:image/...base64..."} -> returns JSON
          * multipart form 'image' -> file upload -> save & render classify.html (legacy)
    Any unexpected exception is logged to classify_error.log.
    """
    prediction = None
    videos = []
    uploaded_image = None
    confidence = None

    if request.method == "POST":
        try:
            # -----------------------
            # 1) Webcam frame (multipart/form-data 'frame')
            # -----------------------
            if "frame" in request.files:
                try:
                    frame_file = request.files["frame"]

                    if frame_file is None:
                        return jsonify({"error": "no_frame", "message": "No 'frame' part in request"}), 400

                    frame_bytes = frame_file.read()
                    if not frame_bytes or len(frame_bytes) < 120:
                        return jsonify({"error": "empty_or_too_small", "message": "Frame empty or too small"}), 400

                    # Validate image bytes using PIL
                    try:
                        _ = Image.open(BytesIO(frame_bytes)).convert("RGB")
                    except Exception as img_err:
                        debug_dir = os.path.join(os.getcwd(), "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = os.path.join(debug_dir, f"frame_bad_{int(time.time())}.jpg")
                        try:
                            with open(debug_path, "wb") as fh:
                                fh.write(frame_bytes)
                        except Exception as save_err:
                            current_app.logger.exception("Failed to save bad frame for debug: %s", save_err)
                        current_app.logger.exception("Invalid image bytes received; saved to %s", debug_path)
                        return jsonify({"error": "invalid_image", "message": "Uploaded frame is not a valid image", "debug_path": debug_path}), 400

                    # Preprocess & classify
                    try:
                        arr = _preprocess_from_bytes(frame_bytes)
                    except Exception as pre_e:
                        current_app.logger.exception("Preprocessing failed for webcam frame: %s", pre_e)
                        return jsonify({"error": "preprocess_failed", "message": str(pre_e)}), 500

                    try:
                        label, conf, idx, topk = classify_array(arr, prefer_tflite=True, return_top_k=3)
                    except Exception as pred_e:
                        current_app.logger.exception("Prediction failed for webcam frame: %s", pred_e)
                        debug_dir = os.path.join(os.getcwd(), "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        dbg_path = os.path.join(debug_dir, f"frame_pred_error_{int(time.time())}.jpg")
                        try:
                            with open(dbg_path, "wb") as fh:
                                fh.write(frame_bytes)
                        except Exception as save_err:
                            current_app.logger.exception("Failed to save frame on prediction error: %s", save_err)
                        return jsonify({
                            "error": "prediction_failed",
                            "message": str(pred_e),
                            "debug_path": dbg_path
                        }), 500

                    # Normalize label to disposal_info keys if needed
                    if label not in disposal_info:
                        normalized = str(label).strip().lower()
                        if normalized in ("o", "organic"):
                            label = "Organic"
                        elif normalized in ("r", "recyclable"):
                            label = "Recyclable"
                        elif "e" in normalized or "ewaste" in normalized or "e-waste" in normalized:
                            label = "E-Waste"
                        elif "hazard" in normalized:
                            label = "Hazardous"
                        else:
                            label = "Other"

                    return jsonify({
                        "label": label,
                        "confidence": float(conf),
                        "idx": int(idx),
                        "top3": topk
                    }), 200

                except Exception as e:
                    current_app.logger.exception("Unexpected exception handling webcam frame: %s", e)
                    return jsonify({"error": "internal_error", "message": str(e)}), 500

            # -----------------------
            # 2) JSON base64 frame (body JSON: { "frame": "data:image/...base64..." } or { "image": base64 })
            # -----------------------
            try:
                data = request.get_json(silent=True)
            except Exception:
                data = None

            if data and (("frame" in data) or ("image" in data)):
                try:
                    b64 = data.get("frame") or data.get("image")
                    if isinstance(b64, str) and b64.startswith("data:"):
                        b64 = b64.split(",", 1)[1]
                    frame_bytes = base64.b64decode(b64)
                    if not frame_bytes or len(frame_bytes) < 120:
                        return jsonify({"error": "empty_or_too_small", "message": "Frame empty or too small"}), 400

                    # Validate
                    try:
                        _ = Image.open(BytesIO(frame_bytes)).convert("RGB")
                    except Exception as img_err:
                        debug_dir = os.path.join(os.getcwd(), "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = os.path.join(debug_dir, f"frame_bad_{int(time.time())}.jpg")
                        try:
                            with open(debug_path, "wb") as fh:
                                fh.write(frame_bytes)
                        except Exception as save_err:
                            current_app.logger.exception("Failed to save bad json-frame for debug: %s", save_err)
                        current_app.logger.exception("Invalid JSON image bytes received; saved to %s", debug_path)
                        return jsonify({"error": "invalid_image", "message": "Uploaded JSON frame not a valid image", "debug_path": debug_path}), 400

                    arr = _preprocess_from_bytes(frame_bytes)
                    label, conf, idx, topk = classify_array(arr, prefer_tflite=True, return_top_k=3)

                    if label not in disposal_info:
                        normalized = str(label).strip().lower()
                        if normalized in ("o", "organic"):
                            label = "Organic"
                        elif normalized in ("r", "recyclable"):
                            label = "Recyclable"
                        elif "e" in normalized or "ewaste" in normalized or "e-waste" in normalized:
                            label = "E-Waste"
                        elif "hazard" in normalized:
                            label = "Hazardous"
                        else:
                            label = "Other"

                    return jsonify({
                        "label": label,
                        "confidence": float(conf),
                        "idx": int(idx),
                        "top3": topk
                    }), 200

                except Exception as e:
                    current_app.logger.exception("JSON-frame handling failed: %s", e)
                    return jsonify({"error": "json_frame_error", "message": str(e)}), 500

            # -----------------------
            # 3) Legacy file upload (form field 'image') -> save file & render page
            # -----------------------
            img = request.files.get("image")
            if img and img.filename:
                try:
                    from werkzeug.utils import secure_filename
                    import time as _time
                    os.makedirs("static/uploads", exist_ok=True)
                    filename = secure_filename(img.filename)
                    unique_filename = f"{int(_time.time())}_{filename}"
                    img_path = os.path.join("static", "uploads", unique_filename)
                    img.save(img_path)
                    uploaded_image = url_for('static', filename=f'uploads/{unique_filename}')

                    try:
                        arr = _preprocess_from_path(img_path)
                        label, conf, idx, topk = classify_array(arr, prefer_tflite=True, return_top_k=3)
                        if label not in disposal_info:
                            normalized = str(label).strip().lower()
                            if normalized in ("o", "organic"):
                                label = "Organic"
                            elif normalized in ("r", "recyclable"):
                                label = "Recyclable"
                            elif "e" in normalized or "ewaste" in normalized or "e-waste" in normalized:
                                label = "E-Waste"
                            elif "hazard" in normalized:
                                label = "Hazardous"
                            else:
                                label = "Other"
                        prediction = label
                        confidence = round(conf, 3)
                    except Exception:
                        current_app.logger.exception("Classification pipeline failed, falling back to predict_image")
                        try:
                            prediction = predict_image(img_path)
                            if prediction not in disposal_info:
                                prediction = "Other"
                        except Exception:
                            current_app.logger.exception("Fallback predict_image failed")
                            prediction = "Other"
                        confidence = None

                    videos = disposal_info.get(prediction, {}).get("videos", [])
                    return render_template("classify.html", prediction=prediction, videos=videos, uploaded_image=uploaded_image, confidence=confidence)
                except Exception as e:
                    current_app.logger.exception("classify (upload) exception: %s", e)
                    return render_template("classify.html", prediction="Error", videos=[], uploaded_image=None, confidence=None, error=str(e))

            # If none of the above branches matched, return bad request
            return jsonify({"error": "no valid image/frame found in request"}), 400

        except Exception as ex:
            # write full traceback to classify_error.log for inspection
            import traceback
            tb = traceback.format_exc()
            try:
                with open("classify_error.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n\n=== {datetime.datetime.utcnow().isoformat()} ===\n")
                    fh.write(tb)
            except Exception as write_err:
                current_app.logger.exception("failed to write classify_error.log: %s", write_err)

            current_app.logger.exception("Unhandled exception in classify POST")

            return jsonify({
                "error": "internal server error in classify POST",
                "message": str(ex),
                "note": "full traceback written to classify_error.log"
            }), 500

    # GET -> render template normally
    return render_template("classify.html", prediction=None, videos=[], uploaded_image=None, confidence=None)

@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    forecast = None
    area = ""
    days = None
    avg_forecast = None
    labels = []
    data = []
    error = None

    if request.method == "POST":
        try:
            area = (request.form.get("area") or "").strip()
            days_raw = request.form.get("days")
            if days_raw is None or area == "":
                raise ValueError("Please provide both 'area' and 'days'.")
            days = int(days_raw)

            if os.path.exists(MODEL_FORECAST_PATH) and os.path.exists(AREA_ENCODER_PATH):
                try:
                    model_local = joblib.load(MODEL_FORECAST_PATH)
                    encoder_local = joblib.load(AREA_ENCODER_PATH)
                    if area in getattr(encoder_local, "classes_", []):
                        encoded_area = int(encoder_local.transform([area])[0])
                        X_input = np.array([[encoded_area, d] for d in range(1, days + 1)])
                        y_pred = model_local.predict(X_input).astype(float).flatten()
                        forecast = float(np.sum(y_pred))
                        avg_forecast = float(np.mean(y_pred)) if y_pred.size > 0 else None
                        labels = [f"Day {i}" for i in range(1, days + 1)]
                        data = [float(x) for x in y_pred.tolist()]
                        return render_template("forecast.html", forecast=round(forecast, 2), area=area, days=days,
                                               avg_forecast=round(avg_forecast, 2) if avg_forecast is not None else None,
                                               labels=labels, data=data, error=None)
                    else:
                        error = f"Area '{area}' not found in model encoder. Using dataset fallback."
                except Exception as e:
                    error = f"Failed to use pre-trained model: {e}. Using dataset fallback."

            csv_path = FORECAST_CSV_PRIMARY if os.path.exists(FORECAST_CSV_PRIMARY) else (FORECAST_CSV_ALT if os.path.exists(FORECAST_CSV_ALT) else None)
            if not csv_path:
                raise FileNotFoundError("No dataset file found for fallback forecasting.")

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            if not {'year', 'area', 'waste_generated'}.issubset(df.columns):
                if 'waste_amount' in df.columns:
                    df = df.rename(columns={'waste_amount': 'waste_generated'})
                else:
                    waste_data = []
                    recycle_summary = {}
                    trend_data = []
                    top_areas = []
                    return render_template("dashboard.html",
                                           waste_data=waste_data,
                                           recycle_data=recycle_summary,
                                           trend_data=trend_data,
                                           top_areas=top_areas,
                                           sensor_summary=[],
                                           error="Dataset does not contain required columns.")
            df['area'] = df['area'].astype(str).str.strip()
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['waste_generated'] = pd.to_numeric(df['waste_generated'], errors='coerce').fillna(0)

            if area not in df['area'].unique():
                raise ValueError(f"Area '{area}' not found in dataset for fallback forecasting.")

            area_df = df[df['area'] == area].sort_values('year')
            X_years = area_df['year'].values.reshape(-1, 1)
            y_values = area_df['waste_generated'].values

            if len(X_years) < 2:
                avg_per_year = float(np.mean(y_values)) if len(y_values) > 0 else 0.0
                y_pred = np.array([avg_per_year] * days, dtype=float)
                labels = [f"Year {i}" for i in range(1, days + 1)]
                data = [float(x) for x in y_pred.tolist()]
                forecast = float(np.sum(y_pred))
                avg_forecast = float(np.mean(y_pred))
            else:
                reg = LinearRegression()
                reg.fit(X_years, y_values)
                last_year = int(np.nanmax(area_df['year'].values))
                future_years = np.arange(last_year + 1, last_year + 1 + days).reshape(-1, 1)
                y_pred = reg.predict(future_years).astype(float)
                labels = [str(int(y[0])) for y in future_years]
                data = [float(max(0.0, x)) for x in y_pred.tolist()]
                forecast = float(np.sum(data))
                avg_forecast = float(np.mean(data))
            return render_template("forecast.html", forecast=round(forecast, 2), area=area, days=days,
                                   avg_forecast=round(avg_forecast, 2) if avg_forecast is not None else None,
                                   labels=labels, data=data, error=error)
        except Exception as e:
            error = str(e)
            return render_template("forecast.html", forecast=None, area=area, days=days, avg_forecast=None, labels=[], data=[], error=error)

    return render_template("forecast.html", forecast=None, area="", days=None, avg_forecast=None, labels=[], data=[], error=None)

# ----------------------------
# Sensor receiver endpoint
# (same as original - unchanged)
# ----------------------------
@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            data = request.form.to_dict()

        ts = data.get("timestamp") or data.get("time") or None
        if ts:
            try:
                ts = pd.to_datetime(ts).isoformat()
            except:
                ts = pd.Timestamp.utcnow().isoformat()
        else:
            ts = pd.Timestamp.utcnow().isoformat()

        row = {
            "timestamp": ts,
            "node_id": data.get("node_id") or data.get("device_id") or data.get("id") or "node_unknown",
            "area": data.get("area", ""),
            "lat": data.get("lat") or data.get("latitude") or "",
            "lon": data.get("lon") or data.get("longitude") or "",
            "fill_level_pct": data.get("fill_level_pct") or data.get("level_percent") or 0,
            "temperature_c": data.get("temperature_c") or data.get("temp") or "",
            "battery_v": data.get("battery_v") or "",
            "sensor_status": data.get("sensor_status") or "OK"
        }

        file_exists = os.path.exists(LIVE_SENSOR_CSV)
        with open(LIVE_SENSOR_CSV, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        file_exists2 = os.path.exists(SENSOR_AGG_CSV)
        with open(SENSOR_AGG_CSV, "a", newline="") as csvfile2:
            writer2 = csv.writer(csvfile2)
            if not file_exists2:
                writer2.writerow(["timestamp", "device_id", "area", "level_percent", "lat", "lon", "weight_kg"])
            writer2.writerow([row["timestamp"], row["node_id"], row["area"], row["fill_level_pct"], row["lat"], row["lon"], ""])

        try:
            area_safe = (row.get("area") or "unknown").strip()
            room_name = f"area_{area_safe.replace(' ', '_')}"
            emit_payload = {
                "node_id": row.get("node_id"),
                "area": row.get("area"),
                "lat": float(row.get("lat")) if str(row.get("lat")) != "" else None,
                "lon": float(row.get("lon")) if str(row.get("lon")) != "" else None,
                "fill_level_pct": float(row.get("fill_level_pct") or 0),
                "temperature_c": row.get("temperature_c"),
                "battery_v": row.get("battery_v"),
                "timestamp": row.get("timestamp"),
                "sensor_status": row.get("sensor_status")
            }
            socketio.emit('sensor_update', emit_payload, room=room_name)
            socketio.emit('sensor_update', emit_payload, room='admin')
        except Exception as e:
            print("SocketIO emit error in /api/sensor:", e)

        return jsonify({"status": "ok", "row": row}), 201
    except Exception as e:
        print("Error in /api/sensor:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/sensor/latest_summary")
def api_sensor_latest_summary():
    try:
        summary = SENSOR_CACHE.get("summary") or []
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sensor/history")
def api_sensor_history():
    node_id = request.args.get("node_id")
    limit = int(request.args.get("limit", 200))
    try:
        df = load_sensor_data()
        if node_id:
            df_node = df[df['node_id'] == node_id].sort_values('timestamp', ascending=True)
        else:
            df_node = df.sort_values('timestamp', ascending=True)
        if df_node.empty:
            return jsonify([])
        out = df_node.tail(limit)[['timestamp', 'node_id', 'fill_level_pct', 'lat', 'lon']].copy()
        out['timestamp'] = out['timestamp'].astype(str)
        return jsonify(out.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Dashboard (kept original)
# ----------------------------
def downsample_series(df_series, max_points=120):
    if len(df_series) <= max_points:
        return df_series
    step = max(1, int(len(df_series) / max_points))
    return df_series.iloc[::step]

@app.route("/dashboard")
def dashboard():
    try:
        sensor_summary = SENSOR_CACHE.get("summary", [])

        csv_path = FORECAST_CSV_PRIMARY if os.path.exists(FORECAST_CSV_PRIMARY) else (FORECAST_CSV_ALT if os.path.exists(FORECAST_CSV_ALT) else None)
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            if not {'year', 'area', 'waste_generated'}.issubset(df.columns):
                if 'waste_amount' in df.columns:
                    df = df.rename(columns={'waste_amount': 'waste_generated'})
                else:
                    waste_data = []
                    recycle_summary = {}
                    trend_data = []
                    top_areas = []
                    return render_template("dashboard.html",
                                           waste_data=waste_data,
                                           recycle_data=recycle_summary,
                                           trend_data=trend_data,
                                           top_areas=top_areas,
                                           sensor_summary=sensor_summary,
                                           error="Dataset does not contain required columns.")
            df['area'] = df['area'].astype(str).str.strip()
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['waste_generated'] = pd.to_numeric(df['waste_generated'], errors='coerce').fillna(0)

            waste_summary = df.groupby("area")["waste_generated"].sum().reset_index().rename(columns={"waste_generated": "total_waste"})
            waste_data = waste_summary.to_dict(orient="records")

            if 'type' in df.columns:
                recycle_summary = df['type'].value_counts().to_dict()
            else:
                recycle_summary = {"Recyclable": 60, "Non-Recyclable": 40}

            trend_summary = df.groupby("year")["waste_generated"].sum().reset_index().sort_values("year")
            if len(trend_summary) >= 2:
                X = trend_summary["year"].values.reshape(-1, 1)
                y = trend_summary["waste_generated"].values
                reg = LinearRegression()
                reg.fit(X, y)
                future_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
                y_pred = reg.predict(future_years)
                future_df = pd.DataFrame({"year": future_years.flatten(), "waste_generated": y_pred})
                full_trend = pd.concat([trend_summary, future_df], ignore_index=True)
                full_trend = full_trend.sort_values("year")
                full_trend_small = downsample_series(full_trend, max_points=120)
                trend_data = full_trend_small.to_dict(orient="records")
            else:
                trend_data = trend_summary.to_dict(orient="records")

            top_areas = waste_summary.sort_values(by="total_waste", ascending=False).head(5).to_dict(orient="records")
        else:
            waste_data = []
            recycle_summary = {}
            trend_data = []
            top_areas = []

        return render_template("dashboard.html",
                               waste_data=waste_data,
                               recycle_data=recycle_summary,
                               trend_data=trend_data,
                               top_areas=top_areas,
                               sensor_summary=sensor_summary,
                               error=None,
                               current_user=session.get('username'),
                               current_role=session.get('role'),
                               assigned_areas=session.get('assigned_areas', []))
    except Exception as e:
        print("❌ Dashboard error:", e)
        return render_template("dashboard.html",
                               waste_data=[],
                               recycle_data={},
                               trend_data=[],
                               top_areas=[],
                               sensor_summary=[],
                               error=str(e),
                               current_user=session.get('username'),
                               current_role=session.get('role'),
                               assigned_areas=session.get('assigned_areas', []))

# ----------------------------
# ORS endpoints (unchanged)
# ----------------------------
@app.route("/optimize_route", methods=["GET", "POST"])
def optimize_route():
    optimized_data = None
    if request.method == "POST":
        try:
            points = request.json.get("points")
            if not points or len(points) < 2:
                return jsonify({"error": "Please provide at least 2 points."}), 400
            url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
            headers = {"Authorization": OPENROUTESERVICE_API_KEY, "Content-Type": "application/json"}
            payload = {"coordinates": points, "optimize_waypoints": True}
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            if "features" in data:
                optimized_data = data
            else:
                return jsonify({"error": "No valid route found", "details": data}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template("optimize_route.html", data=optimized_data)

@app.route("/api/optimize_multi_truck", methods=["POST"])
def optimize_multi_truck():
    try:
        payload = request.get_json(force=True)
        collection_points = payload.get("collection_points", [])
        vehicle_count = int(payload.get("vehicle_count", 1))
        if len(collection_points) == 0:
            return jsonify({"error": "No collection_points provided"}), 400
        if vehicle_count < 1:
            return jsonify({"error": "vehicle_count must be >=1"}), 400

        depot_lat = 12.9716
        depot_lng = 77.5946

        jobs = [{"id": idx + 1, "location": [float(p["lng"]), float(p["lat"])]}
                for idx, p in enumerate(collection_points)]
        vehicles = [{"id": v + 1, "start": [depot_lng, depot_lat], "end": [depot_lng, depot_lat]}
                    for v in range(vehicle_count)]
        optimization_payload = {"jobs": jobs, "vehicles": vehicles, "options": {"g": True}}

        opt_url = "https://api.openrouteservice.org/optimization"
        hdr = {"Authorization": ORS_OPT_KEY, "Content-Type": "application/json"}

        resp = requests.post(opt_url, json=optimization_payload, headers=hdr, timeout=60)
        resp.raise_for_status()
        opt_data = resp.json()

        vehicles_out = []
        routes = opt_data.get("routes", [])
        for route in routes:
            vehicle_id = route.get("vehicle")
            steps = route.get("steps", [])
            coords_for_directions = [[depot_lng, depot_lat]]
            for step in steps:
                if step.get("type") == "job":
                    job_id = step.get("job")
                    job = next((j for j in jobs if j["id"] == job_id), None)
                    if job:
                        coords_for_directions.append(job["location"])
            coords_for_directions.append([depot_lng, depot_lat])

            geojson_route = None
            distance_m = None
            duration_s = None

            if "geometry" in route and route["geometry"]:
                geom = route["geometry"]
                if isinstance(geom, dict) and "coordinates" in geom:
                    geojson_route = {"type": "Feature", "geometry": geom, "properties": {}}

            if geojson_route is None:
                try:
                    directions_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
                    dir_resp = requests.post(directions_url, json={"coordinates": coords_for_directions}, headers=hdr, timeout=60)
                    dir_resp.raise_for_status()
                    dir_data = dir_resp.json()
                    if "features" in dir_data and len(dir_data["features"]) > 0:
                        feature = dir_data["features"][0]
                        geojson_route = feature
                        props = feature.get("properties", {})
                        summary = props.get("summary", {})
                        distance_m = summary.get("distance", 0)
                        duration_s = summary.get("duration", 0)
                    else:
                        geojson_route = None
                        distance_m = 0
                        duration_s = 0
                except Exception as e:
                    print("Directions call failed for vehicle", vehicle_id, e)
                    geojson_route = None
                    distance_m = 0
                    duration_s = 0

            if distance_m is None:
                distance_m = route.get("distance") or route.get("summary", {}).get("distance", 0)
            if duration_s is None:
                duration_s = route.get("duration") or route.get("summary", {}).get("duration", 0)

            vehicles_out.append({
                "vehicle_id": vehicle_id,
                "distance_m": distance_m,
                "duration_s": duration_s,
                "geojson": geojson_route
            })
        return jsonify({"status": "ok", "vehicles": vehicles_out, "raw_ors_response": opt_data})
    except Exception as e:
        print("Error in /api/optimize_multi_truck:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Chatbot (kept original)
# ----------------------------
chatbot_responses = {
    "hello": "Hello! How can I help you with waste management today?",
    "hi": "Hi there! I can guide you about waste disposal, forecasting, and route optimization.",
    "organic": "Organic waste should be composted or put in green bins.",
    "recyclable": "Rinse and sort recyclables before placing in recycling bins.",
    "hazardous": "Hazardous waste should go to designated collection centers.",
    "e-waste": "Electronic waste should be recycled at certified e-waste disposal centers.",
    "forecast": "Use Forecasting to predict future waste.",
    "route": "Use the GIS Map to optimize routes.",
    "dashboard": "The Dashboard shows interactive charts.",
    "thanks": "You're welcome!"
}

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").lower()
    if not user_msg:
        return jsonify({"reply": "Please type a message."})
    reply = "Sorry, I didn't understand that."
    for key, response in chatbot_responses.items():
        if key in user_msg:
            reply = response
            break
    return jsonify({"reply": reply})

@app.route("/gis")
def gis():
    return render_template("gis.html")

# ----------------------------
# SocketIO handlers (kept original)
# ----------------------------
@socketio.on('connect')
def handle_socket_connect():
    try:
        username = session.get('username')
        role = session.get('role')
        assigned = session.get('assigned_areas', []) or []
        if role == 'admin':
            join_room('admin')
            print(f"[socket] admin {username} joined admin room")
        elif role == 'consumer':
            for a in assigned:
                room = f"area_{str(a).replace(' ', '_')}"
                join_room(room)
            print(f"[socket] consumer {username} joined rooms: {assigned}")
        else:
            print("[socket] connection (unauthenticated)")
    except Exception as e:
        print("[socket connect] error:", e)

@socketio.on('disconnect')
def handle_socket_disconnect():
    try:
        username = session.get('username')
        print(f"[socket] disconnected: {username}")
    except:
        pass

# ----------------------------
# Background sensor cache refresher thread (kept original)
# ----------------------------
def sensor_cache_refresher(poll_interval=3):
    while True:
        try:
            df = load_sensor_data(force_reload=True, max_rows_from_history=20000)
            if df is not None and not df.empty:
                last = df.dropna(subset=["timestamp"]).sort_values("timestamp").groupby("node_id").last().reset_index()
                summary = []
                for _, r in last.iterrows():
                    summary.append({
                        "node_id": str(r.get("node_id")),
                        "area": str(r.get("area")) if pd.notna(r.get("area")) else "",
                        "lat": float(r.get("lat")) if pd.notna(r.get("lat")) else None,
                        "lon": float(r.get("lon")) if pd.notna(r.get("lon")) else None,
                        "fill_level_pct": float(r.get("fill_level_pct") or 0),
                        "temperature_c": float(r.get("temperature_c") or 0),
                        "battery_v": float(r.get("battery_v") or 0),
                        "last_seen": pd.to_datetime(r.get("timestamp")).isoformat() if pd.notna(r.get("timestamp")) else None,
                        "sensor_status": r.get("sensor_status") if pd.notna(r.get("sensor_status")) else "OK"
                    })
                SENSOR_CACHE["summary"] = summary
            else:
                SENSOR_CACHE["summary"] = []
            SENSOR_CACHE["last_load"] = pd.Timestamp.utcnow()
        except Exception as e:
            print("Sensor cache refresher error:", e)
        time.sleep(poll_interval)

if __name__ == "__main__":
    # start background refresher
    refresher_thread = threading.Thread(target=sensor_cache_refresher, args=(3,), daemon=True)
    refresher_thread.start()

    # Run socketio server in threading mode (no eventlet)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
