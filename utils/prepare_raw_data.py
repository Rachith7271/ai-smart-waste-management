import pandas as pd
import os

RAW_DIR = "raw_data"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Hardcoded fallback coordinates (Bengaluru city centre)
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

def clean_waste_csv(file_path):
    print(f"Processing {file_path} ...")
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    possible_year_cols = [c for c in df.columns if "year" in c.lower() or "Year" in c]
    possible_waste_cols = [c for c in df.columns if "waste" in c.lower() or "tonne" in c.lower() or "generated" in c.lower()]

    rows = []
    for _, row in df.iterrows():
        # Find year
        year = None
        for ycol in possible_year_cols:
            try:
                year = int(str(row[ycol])[:4])
                break
            except:
                continue
        if not year:
            year = 2017  # fallback if not found

        # Find waste
        waste_kg = None
        for wcol in possible_waste_cols:
            try:
                val = float(str(row[wcol]).replace(',', ''))
                # if it's tonnes, convert to kg
                if val < 10000:  
                    val = val * 1000
                waste_kg = val
                break
            except:
                continue
        if not waste_kg:
            continue

        # Area name (if exists)
        area = row.get('Ward', row.get('Zone', row.get('Area', 'Bengaluru')))

        rows.append({
            "date": f"{year}-01-01",
            "area": area,
            "waste_kg": waste_kg,
            "lat": BENGALURU_LAT,
            "lon": BENGALURU_LON,
            "source": os.path.basename(file_path)
        })
    return rows


def main():
    all_rows = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            fpath = os.path.join(RAW_DIR, file)
            all_rows.extend(clean_waste_csv(fpath))

    if not all_rows:
        print("No usable data found!")
        return

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(OUT_DIR, "cleaned_waste_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\n✅ Saved cleaned dataset to: {out_file}")
    print("Rows:", len(df_out))


if __name__ == "__main__":
    main()
