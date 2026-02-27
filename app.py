import eventlet
eventlet.monkey_patch()
import joblib
from tensorflow.keras.models import load_model
import os
import csv
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room

# ---------------------------------------------------
# BASE DIRECTORIES (Render Safe)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

LIVE_SENSOR_CSV = os.path.join(DATA_DIR, "sensor_live_append.csv")

# ---------------------------------------------------
# FLASK + SOCKETIO
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"

socketio = SocketIO(app, cors_allowed_origins="*")

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


# start background thread safely
socketio.start_background_task(sensor_cache_refresher)

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# ✅ CLASSIFY (SAFE MODE — NO MODEL CRASH)
classification_model = None

def get_model():
    global classification_model
    if classification_model is None:
        try:
            classification_model = load_model(
                os.path.join(BASE_DIR, "models", "waste_classifier.h5"),
                compile=False
            )
            print("✅ AI Model Loaded")
        except Exception as e:
            print("❌ Model load error:", e)
            classification_model = None
    return classification_model


@app.route("/classify", methods=["GET", "POST"])
def classify():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No image uploaded"
        else:
            file = request.files["image"]

            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))

            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            model = get_model()

            if model is None:
                error = "Model not available"
            else:
                preds = model.predict(arr)[0]
                idx = int(np.argmax(preds))
                confidence = float(np.max(preds))

                labels = ["Organic", "Recyclable", "Hazardous", "E-Waste", "Other"]
                prediction = labels[idx]

    return render_template(
        "classify.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# ✅ DASHBOARD (NO TEMPLATE ERRORS)
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        sensor_summary=SENSOR_CACHE.get("summary", []),
        waste_data=[],
        recycle_data={},
        trend_data=[],
        top_areas=[]
    )


# ✅ FORECAST (SAFE VERSION)
import joblib

forecast_model = None
area_encoder = None

def load_forecast_model():
    global forecast_model, area_encoder

    if forecast_model is None:
        try:
            forecast_model = joblib.load(
                os.path.join(BASE_DIR, "models", "waste_forecast_model.pkl")
            )
            area_encoder = joblib.load(
                os.path.join(BASE_DIR, "models", "area_encoder.pkl")
            )
            print("✅ Forecast model loaded")
        except Exception as e:
            print("❌ Forecast model load error:", e)
            forecast_model = None
            area_encoder = None


@app.route("/forecast", methods=["GET", "POST"])
def forecast():

    forecast_value = None
    labels = []
    data = []
    error = None

    if request.method == "POST":

        area = request.form.get("area")
        days = request.form.get("days")

        try:
            days = int(days)
            load_forecast_model()

            if forecast_model is None or area_encoder is None:
                error = "Forecast model not available"
            else:
                encoded_area = area_encoder.transform([area])[0]

                X_input = np.array([[encoded_area, d] for d in range(1, days + 1)])

                y_pred = forecast_model.predict(X_input)

                forecast_value = float(np.sum(y_pred))
                labels = [f"Day {i}" for i in range(1, days + 1)]
                data = y_pred.tolist()

        except Exception as e:
            error = str(e)

    return render_template(
        "forecast.html",
        forecast=forecast_value,
        labels=labels,
        data=data,
        error=error
    )

# ✅ GIS PAGE
@app.route("/gis")
def gis():
    return render_template("gis.html")


# ---------------------------------------------------
# SENSOR API
# ---------------------------------------------------
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
# SOCKET EVENTS
# ---------------------------------------------------
@socketio.on("connect")
def handle_connect():
    join_room("admin")
    print("Client connected")


# ---------------------------------------------------
# RUN (LOCAL ONLY)
# Render uses gunicorn automatically
# ---------------------------------------------------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)