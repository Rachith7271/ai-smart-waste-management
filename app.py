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

from flask import Flask, render_template, request, jsonify, url_for, session
from flask_socketio import SocketIO, emit, join_room

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# BASE DIRECTORIES (Render-safe)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------
LIVE_SENSOR_CSV = os.path.join(DATA_DIR, "sensor_live_append.csv")
MODEL_CLASS_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.h5")
CLASS_LABELS_JSON = os.path.join(BASE_DIR, "models", "class_labels.json")

# ---------------------------------------------------
# FLASK + SOCKETIO (Render compatible)
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"

socketio = SocketIO(
    app,
    async_mode="eventlet",
    cors_allowed_origins="*"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
classification_model = None
LABELS_MAP = {0:"Organic",1:"Recyclable",2:"Hazardous",3:"E-Waste",4:"Other"}

try:
    if os.path.exists(MODEL_CLASS_PATH):
        classification_model = load_model(MODEL_CLASS_PATH)
        print("✅ Classification model loaded.")
except Exception as e:
    print("Model load error:", e)

# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess_image_from_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------------------
# SENSOR CACHE
# ---------------------------------------------------
SENSOR_CACHE = {"summary": []}

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
            "fill_level_pct": float(r.get("fill_level_pct",0)),
            "lat": float(r.get("lat",0)),
            "lon": float(r.get("lon",0))
        })
    return summary

def sensor_cache_refresher():
    while True:
        try:
            SENSOR_CACHE["summary"] = load_sensor_data()
        except Exception as e:
            print("Sensor refresh error:", e)
        time.sleep(3)

# ---------------------------------------------------
# START BACKGROUND THREAD (WORKS ON RENDER)
# ---------------------------------------------------
def start_background_tasks():
    global _bg_started
    if globals().get("_bg_started"):
        return
    _bg_started = True
    print("✅ Starting sensor background thread...")
    t = threading.Thread(target=sensor_cache_refresher, daemon=True)
    t.start()

start_background_tasks()

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["GET","POST"])
def classify():
    if request.method == "POST":
        if "frame" in request.files:
            frame = request.files["frame"]
            frame_bytes = frame.read()
            arr = preprocess_image_from_bytes(frame_bytes)

            preds = classification_model.predict(arr)[0]
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = LABELS_MAP.get(idx,"Other")

            return jsonify({
                "label": label,
                "confidence": confidence
            })
        return jsonify({"error":"No frame provided"}),400

    return render_template("classify.html")

@app.route("/dashboard")
def dashboard():
    sensor_summary = SENSOR_CACHE.get("summary", [])
    return render_template("dashboard.html", sensor_summary=sensor_summary)

@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    data = request.get_json(force=True)
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "node_id": data.get("node_id","node_unknown"),
        "area": data.get("area",""),
        "fill_level_pct": data.get("fill_level_pct",0),
        "lat": data.get("lat",0),
        "lon": data.get("lon",0)
    }

    file_exists = os.path.exists(LIVE_SENSOR_CSV)
    with open(LIVE_SENSOR_CSV,"a",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    socketio.emit("sensor_update", row)

    return jsonify({"status":"ok"})

# ---------------------------------------------------
# SOCKET EVENTS
# ---------------------------------------------------
@socketio.on("connect")
def handle_connect():
    join_room("admin")

# ---------------------------------------------------
# RUN (Render uses PORT env variable)
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)