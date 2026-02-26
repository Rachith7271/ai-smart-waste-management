import pandas as pd

df = pd.read_csv("data/cleaned_waste_data.csv")

# If date column has wrong name, fix it
if 'date' not in df.columns:
    for c in df.columns:
        if 'date' in c.lower():
            df.rename(columns={c: 'date'}, inplace=True)
            break

# Drop blank area rows
df = df.dropna(subset=['area'])

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Group by year, month, area
grouped = df.groupby(['year', 'month', 'area'])['waste_kg'].sum().reset_index()

# Save as processed dataset
grouped.to_csv("data/processed_waste_data.csv", index=False)

print("✅ processed_waste_data.csv created")
