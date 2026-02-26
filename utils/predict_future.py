import pandas as pd
import joblib

# Load model
model = joblib.load("waste_forecast_model.pkl")

# Load dataset to get area encoding reference
df = pd.read_csv("data/cleaned_waste_data.csv")
df['area'] = df['area'].astype(str)
area_codes = dict(zip(df['area'].astype('category').cat.codes, df['area'].unique()))
area_to_code = {v: k for k, v in area_codes.items()}

def predict_waste(area_name, year):
    if area_name not in area_to_code:
        print("⚠️ Area not found in training data. Please use one of these:")
        print(list(area_to_code.keys()))
        return
    area_code = area_to_code[area_name]
    prediction = model.predict([[year, area_code]])
    print(f"📅 Year: {year} | 📍 Area: {area_name}")
    print(f"🔮 Predicted Waste: {prediction[0]:,.0f} kg")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Change these to test
    predict_waste("Bengaluru", 2040)
