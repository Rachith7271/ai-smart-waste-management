import pandas as pd

# Load your raw data
df = pd.read_csv("data/cleaned_waste_data.csv")

# Remove rows with missing area
df = df.dropna(subset=['area'])

# Extract year from date
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

# Rename columns to expected names
df = df.rename(columns={
    'area': 'area',
    'waste_kg': 'waste_generated'
})

# Select only needed columns
df = df[['year', 'area', 'waste_generated']]

# Save cleaned data
df.to_csv("data/cleaned_waste_data.csv", index=False)

print("✅ Cleaned dataset saved to data/cleaned_waste_data.csv")
