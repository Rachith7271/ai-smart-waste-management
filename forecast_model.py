import pandas as pd
from prophet import Prophet
import joblib

# Load processed data
df = pd.read_csv("data/processed_waste_data.csv")

# Train a model for each area
areas = df['area'].unique()
models = {}

for area in areas:
    area_df = df[df['area'] == area][['year','month','waste_kg']].copy()
    area_df['ds'] = pd.to_datetime(area_df['year'].astype(str) + '-' + area_df['month'].astype(str) + '-01')
    area_df['y'] = area_df['waste_kg']

    m = Prophet()
    m.fit(area_df[['ds','y']])
    models[area] = m

# Save all models
joblib.dump(models, "waste_forecast_models.pkl")
print("✅ All area Prophet models saved")
