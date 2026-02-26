import pandas as pd
import joblib
import streamlit as st

model = joblib.load("waste_forecast_model.pkl")
le = joblib.load("area_encoder.pkl")

df = pd.read_csv("data/cleaned_waste_data.csv")
areas = sorted(df['area'].unique())

st.title("🗑️ Waste Generation Forecast")
area_name = st.selectbox("📍 Select Area", areas)
year = st.number_input("📅 Enter Year", min_value=2024, max_value=2050, value=2026, step=1)

if st.button("🔮 Predict Waste"):
    area_code = le.transform([area_name])[0]
    prediction = model.predict([[year, area_code]])[0]
    st.success(f"**Predicted Waste for {area_name} in {year}: {prediction:,.0f} kg**")
