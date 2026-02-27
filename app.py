import eventlet
eventlet.monkey_patch()

import os
import csv
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room

import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# PATHS (Render Safe)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

LIVE_SENSOR_CSV = os.path.join(DATA_DIR, "sensor_live_append.csv")
MODEL_CLASS_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.h5")

# ---------------------------------------------------
# FLASK + SOCKETIO
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"

socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------------------------------------
# LAZY MODEL LOAD (Render Optimized)
# ---------------------------------------------------
classification_model = None

def get_model():
    global classification_model
    if classification_model is None:
        try:
            print("Loading classification model...")
            classification_model = load_model(
                MODEL_CLASS_PATH,
                compile=False   # IMPORTANT
            )
            print("✅ Model loaded")
        except Exception as e:
            print("⚠ Model load failed:", e)
            classification_model = None
    return classification_model

LABELS_MAP = {
    0: "Organic",
    1: "Recyclable",
    2: "Hazardous",
    3: "E-Waste",
    4: "Other"
}

# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess_image_from_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------
# SENSOR CACHE
# ---------------------------------------------------
SENSOR_CACHE = {"summary": []}
_bg_started = False

def load_sensor_data():
    if not os.path.exists(LIVE_SENSOR_CSV):
        return []

    df = pd.read_csv(LIVE_SENSOR_CSV)
    if df.empty:
        return []

    last = df.sort_values("timestamp").groupby("node_id").last().reset_index()

    summary = []
    for _, r in last.iterrows():
        summary.append({
            "node_id": r.get("node_id"),
            "area": r.get("area"),
            "fill_level_pct": float(r.get("fill_level_pct", 0)),
            "lat": float(r.get("lat", 0)),
            "lon": float(r.get("lon", 0))
        })
    return summary

def sensor_cache_refresher():
    while True:
        try:
            SENSOR_CACHE["summary"] = load_sensor_data()
        except Exception as e:
            print("Sensor refresh error:", e)

        eventlet.sleep(3)

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/forecast")
def forecast():
    return render_template("forecast.html")
@app.route("/gis")
def gis():
    return render_template("gis.html")

@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        sensor_summary=SENSOR_CACHE.get("summary", [])
    )

@app.route("/classify", methods=["GET", "POST"])
def classify():

    if request.method == "POST":

        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        frame_bytes = request.files["frame"].read()
        arr = preprocess_image_from_bytes(frame_bytes)

        model = get_model()
        if model is None:
            return jsonify({"error": "Model not available"}), 500

        preds = model.predict(arr)[0]

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = LABELS_MAP.get(idx, "Other")

        return jsonify({
            "label": label,
            "confidence": confidence
        })

    return render_template("classify.html")

@app.route("/api/sensor", methods=["POST"])
def api_sensor():

    data = request.get_json(force=True)

    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "node_id": data.get("node_id", "node_unknown"),
        "area": data.get("area", ""),
        "fill_level_pct": data.get("fill_level_pct", 0),
        "lat": data.get("lat", 0),
        "lon": data.get("lon", 0)
    }

    file_exists = os.path.exists(LIVE_SENSOR_CSV)

    with open(LIVE_SENSOR_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    socketio.emit("sensor_update", row)
    return jsonify({"status": "ok"})

# ---------------------------------------------------
# SOCKET EVENTS (SAFE BACKGROUND START)
# ---------------------------------------------------
@socketio.on("connect")
def handle_connect():
    global _bg_started

    join_room("admin")

    if not _bg_started:
        print("✅ Starting background sensor refresher")
        socketio.start_background_task(sensor_cache_refresher)
        _bg_started = True

# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)