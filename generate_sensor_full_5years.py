# generate_sensor_full_5years.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, math

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

device_ids = [f"BIN_{area.replace(' ','').replace('.','')}_01" for area in areas]

start = datetime(2020,1,1,0,0,0)
end = datetime(2024,12,31,23,0,0)
hours = int((end - start).total_seconds() // 3600) + 1  # inclusive
print("Total hours:", hours)

# We'll stream to CSV in chunks to avoid huge memory usage
chunk_size = 100000  # rows per write
out_file = "sensor_full_5years_hourly_20sensors.csv"
cols = ["timestamp","device_id","area","latitude","longitude","fill_level_percent","weight_kg"]

np.random.seed(42)
rows_written = 0
with open(out_file, "w") as f:
    # write header
    f.write(",".join(cols) + "\n")
    for i in range(hours):
        ts = start + timedelta(hours=i)
        # for each sensor produce one row
        for s_idx, device in enumerate(device_ids):
            area = areas[s_idx]
            lat, lon = coords[area]
            day_of_year = ts.timetuple().tm_yday
            seasonal = 10 * math.sin(2 * math.pi * day_of_year / 365.0)
            base = (s_idx % 5) * 5 + 20
            noise = np.random.normal(0, 3)
            level = base + seasonal + noise + (i % 24) * 0.2
            level = max(0, min(100, level))
            weight = round((level / 100.0) * 50.0 + np.random.normal(0, 0.5), 2)
            row = [ts.strftime("%Y-%m-%dT%H:%M:%SZ"), device, area, lat, lon, round(level,2), weight]
            f.write(",".join(map(str,row)) + "\n")
            rows_written += 1
        if rows_written % 100000 == 0:
            print("Rows written:", rows_written)

print("Done. File:", out_file)
with open("sensor_full_meta.json","w") as f:
    json.dump({"device_ids": device_ids, "coords": coords, "areas": areas, "rows": rows_written}, f, indent=2)
