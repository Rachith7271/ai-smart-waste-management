# sensor_replay.py
"""
Replay a CSV file (timestamp ordered) into /api/sensor to simulate historical flow.
CSV must contain: timestamp,node_id,area,latitude,longitude,fill_level_percent
Usage:
 python sensor_replay.py --file data/sensor_full_5years_hourly.csv --host http://127.0.0.1:5000 --rate 50
"""
import argparse, csv, requests, time, datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:5000")
    parser.add_argument("--rate", type=float, default=10.0, help="rows per second")
    args = parser.parse_args()

    delay = 1.0 / max(0.1, args.rate)
    with open(args.file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Replaying {len(rows)} rows at {args.rate} rows/s...")
    for r in rows:
        payload = {
            "timestamp": r.get("timestamp"),
            "node_id": r.get("node_id") or r.get("device_id"),
            "area": r.get("area"),
            "lat": r.get("latitude") or r.get("lat"),
            "lon": r.get("longitude") or r.get("lon"),
            "fill_level_pct": r.get("fill_level_percent") or r.get("fill") or r.get("fill_level_pct")
        }
        try:
            requests.post(args.host.rstrip('/') + "/api/sensor", json=payload, timeout=6)
        except Exception as e:
            print("Post error:", e)
        time.sleep(delay)

if __name__ == "__main__":
    main()
