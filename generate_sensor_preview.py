# generate_sensor_preview.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

areas = [
"Koramangala","Indiranagar","Whitefield","Jayanagar","Electronic City","HSR Layout",
"Malleswaram","Basavanagudi","Sarjapur Road","Rajajinagar","Marathahalli","Bellandur",
"Banashankari","Sadashivanagar","Hebbal","BTM Layout","Ulsoor","Frazer Town","J.P. Nagar","Devanahalli"
]

coords = {
"Koramangala": (12.9352,77.6245),"Indiranagar": (12.9719,77.6380),"Whitefield": (12.9699,77.7490),
"Jayanagar": (12.9250,77.5938),"Electronic City": (12.8444,77.6601),"HSR Layout": (12.9180,77.6229),
"Malleswaram": (13.0067,77.5709),"Basavanagudi": (12.9485,77.5740),"Sarjapur Road": (12.9141,77.7198),
"Rajajinagar": (13.0010,77.5480),"Marathahalli": (12.9591,77.6979),"Bellandur": (12.9352,77.6784),
"Banashankari": (12.9172,77.5636),"Sadashivanagar": (13.0046,77.5739),"Hebbal": (13.0358,77.5970),
"BTM Layout": (12.9165,77.6101),"Ulsoor": (12.9775,77.6168),"Frazer Town": (12.9941,77.6192),
"J.P. Nagar": (12.9151,77.5745),"Devanahalli": (13.1986,77.7090)
}

# area-based device ids (choice B)
device_ids = [f"BIN_{area.replace(' ','').replace('.','')}_01" for area in areas]

# timestamps
start = datetime(2020,1,1,0,0,0)
rows_count = 2000
timestamps = [start + timedelta(hours=i) for i in range(rows_count)]

rows = []
np.random.seed(42)
for i, ts in enumerate(timestamps):
    idx = i % len(device_ids)
    device = device_ids[idx]
    area = areas[idx]
    lat, lon = coords[area]
    day_of_year = ts.timetuple().tm_yday
    seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365.0)
    base = (idx % 5) * 5 + 20
    noise = np.random.normal(0, 3)
    level = base + seasonal + noise + (i % 24) * 0.2
    level = max(0, min(100, level))
    weight = round((level / 100.0) * 50.0 + np.random.normal(0, 0.5), 2)
    rows.append({
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_id": device,
        "area": area,
        "latitude": lat,
        "longitude": lon,
        "fill_level_percent": round(level, 2),
        "weight_kg": weight
    })

df = pd.DataFrame(rows)
df.to_csv("sensor_preview_2000rows.csv", index=False)
with open("sensor_preview_meta.json", "w") as f:
    json.dump({"device_ids": device_ids, "coords": coords, "areas": areas}, f, indent=2)

print("Preview saved -> sensor_preview_2000rows.csv")
