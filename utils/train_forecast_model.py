import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np

# Load data
df = pd.read_csv("data/cleaned_waste_data.csv")

# Encode area names
le = LabelEncoder()
df['area_code'] = le.fit_transform(df['area'])

# Features and target
X = df[['year', 'area_code']]
y = df['waste_generated']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Save model and label encoder
joblib.dump(model, "waste_forecast_model.pkl")
joblib.dump(le, "area_encoder.pkl")
print("✅ Model and encoder saved.")
