# seed_sensor_data.py
import requests, random, time
from datetime import datetime, timedelta
areas = ["Koramangala","Indiranagar","Whitefield","Jayanagar","Electronic City","HSR Layout"]
host = "http://127.0.0.1:5000"
endpoint = host + "/api/sensor"

# generate last 30 days sparse data per area
for area in areas:
    base = random.uniform(5,30)
    for d in range(30):
        dt = datetime.utcnow() - timedelta(days=30-d, hours=random.randint(0,23))
        level = max(0, min(100, base + d*random.uniform(0.3,1.2) + random.gauss(0,2)))
        payload = {
            "device_id": f"seed_{area.lower().replace(' ','_')}",
            "area": area,
            "timestamp": dt.isoformat(),
            "distance_cm": 100 - level,
            "level_percent": round(level,2),
            "weight_kg": round(level*0.08 + random.uniform(-2,2),2),
            "battery_v": round(random.uniform(3.7,4.1),2),
            "lat": 12.97,
            "lon": 77.59
        }
        try:
            r = requests.post(endpoint, json=payload, timeout=5)
            print(f"seed {area} {d+1}/30 -> {r.status_code}")
        except Exception as e:
            print("err", e)
        time.sleep(0.05)
