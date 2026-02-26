from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
models = joblib.load("waste_forecast_models.pkl")

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.json
    area = data['area']
    months = int(data.get('months', 12))  # default 12 months

    if area not in models:
        return jsonify({"error": "Area not found"}), 400

    future = models[area].make_future_dataframe(periods=months, freq='M')
    forecast = models[area].predict(future)

    result = forecast[['ds','yhat']].tail(months).to_dict(orient='records')
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # different port from app.py
