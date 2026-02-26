# sensor_simulator.py
"""
Sensor simulator with overflow alerts and truck auto-assignment.

Usage:
  python sensor_simulator.py --host http://127.0.0.1:5000 --interval 10 --nodes 8 --trucks 4

Features:
 - Simulates N sensor nodes around Bengaluru (same areas as before).
 - Sends sensor POST to /api/sensor.
 - When fill_level_pct >= OVERFLOW_THRESHOLD, sends an alert to /api/alert (or /api/alerts)
   and triggers truck assignment (POST to /api/assign_truck or /api/dispatch).
 - Maintains a simulated fleet of trucks, finds nearest idle truck, assigns it, and marks it as busy for SERVICE_DURATION seconds.
 - Prints actions and failures to stdout.
"""
import argparse
import time
import random
import requests
import math
import datetime
import threading

DEFAULT_HOST = "http://127.0.0.1:5000"

AREAS_SAMPLE = [
  ("Koramangala", 12.9352, 77.6245),
  ("Indiranagar", 12.9719, 77.6380),
  ("Whitefield", 12.9699, 77.7490),
  ("Jayanagar", 12.9250, 77.5938),
  ("Electronic City", 12.8444, 77.6601),
  ("HSR Layout", 12.9180, 77.6229),
  ("Malleswaram", 13.0067, 77.5709),
  ("Banashankari", 12.9212, 77.5435)
]

# ----- CONFIG -----
OVERFLOW_THRESHOLD = 85.0   # percent: when to trigger overflow alert & assignment
CRITICAL_THRESHOLD = 95.0   # optional: mark as critical if above this
SERVICE_DURATION = 300      # seconds a truck is 'busy' after being assigned (simulated)
TRUCK_SPEED_KMPH = 25.0     # assumed average speed for ETA estimation
ASSIGNMENT_RETRY = 1        # number of times to retry assignment post if fails
# -------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in kilometers between two lat/lon pairs
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def make_nodes(n):
    nodes=[]
    for i in range(n):
        area, lat, lon = AREAS_SAMPLE[i % len(AREAS_SAMPLE)]
        node_id = f"SIM_{area.replace(' ','_')}_{i+1:02d}"
        # start with random fill
        fill = random.uniform(10, 40)
        nodes.append({
            "node_id": node_id,
            "area": area,
            "lat": lat + random.uniform(-0.002,0.002),
            "lon": lon + random.uniform(-0.002,0.002),
            "fill": fill,
            "trend": random.choice([1,1,1,-0.2])
        })
    return nodes

def make_trucks(n):
    # create a few trucks positioned roughly around Bengaluru center with small random offsets
    center_lat, center_lon = 12.9719, 77.5946  # approximate Bengaluru center
    trucks = []
    for i in range(n):
        tid = f"TRUCK_{i+1:02d}"
        trucks.append({
            "truck_id": tid,
            "lat": center_lat + random.uniform(-0.04, 0.04),
            "lon": center_lon + random.uniform(-0.04, 0.04),
            "status": "idle",       # idle | assigned
            "assigned_at": None,    # timestamp when assigned
            "assigned_node": None
        })
    return trucks

class TruckFleet:
    def __init__(self, trucks):
        self.trucks = trucks
        self.lock = threading.Lock()

    def find_nearest_idle(self, lat, lon):
        with self.lock:
            idle_trucks = [t for t in self.trucks if t["status"] == "idle"]
            if not idle_trucks:
                return None
            nearest = min(idle_trucks, key=lambda t: haversine_km(lat, lon, t["lat"], t["lon"]))
            return nearest

    def mark_assigned(self, truck_id, node_id):
        with self.lock:
            for t in self.trucks:
                if t["truck_id"] == truck_id:
                    t["status"] = "assigned"
                    t["assigned_at"] = time.time()
                    t["assigned_node"] = node_id
                    return True
        return False

    def release_after(self, truck_id, delay_seconds):
        # background timer to free the truck after delay_seconds
        def _release():
            time.sleep(delay_seconds)
            with self.lock:
                for t in self.trucks:
                    if t["truck_id"] == truck_id:
                        t["status"] = "idle"
                        t["assigned_at"] = None
                        t["assigned_node"] = None
                        print(f"[TRUCK] {truck_id} is now idle again (service completed).")
                        break
        thread = threading.Thread(target=_release, daemon=True)
        thread.start()

    def get_snapshot(self):
        with self.lock:
            return [t.copy() for t in self.trucks]

def step_node(node):
    # fill increases by trend +/- noise, occasionally emptied
    if random.random() < 0.02:  # occasional empty event (collection)
        node["fill"] = random.uniform(0,10)
    else:
        node["fill"] = node["fill"] + node["trend"]*random.uniform(0.2,2.5) + random.uniform(-0.5,0.7)
    node["fill"] = max(0.0, min(100.0, node["fill"]))
    return node

def post_sensor(host, node):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "node_id": node["node_id"],
        "area": node["area"],
        "lat": round(node["lat"],6),
        "lon": round(node["lon"],6),
        "fill_level_pct": round(node["fill"],2),
        "temperature_c": round(random.uniform(20,35),2),
        "battery_v": round(random.uniform(3.2,4.2),2)
    }
    try:
        r = requests.post(host.rstrip('/') + "/api/sensor", json=payload, timeout=6)
        if r.status_code in (200,201):
            print(f"[OK] {node['node_id']} -> {payload['fill_level_pct']}%")
        else:
            print("[ERR] /api/sensor", r.status_code, r.text)
    except Exception as e:
        print("[ERR] Post failed to /api/sensor:", e)
    return payload

def send_alert(host, node, level_pct):
    # try a couple of common alert endpoints, be tolerant if backend differs
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "node_id": node["node_id"],
        "area": node["area"],
        "lat": round(node["lat"],6),
        "lon": round(node["lon"],6),
        "fill_level_pct": round(level_pct,2),
        "alert_type": "OVERFLOW" if level_pct >= OVERFLOW_THRESHOLD else "HIGH_FILL",
        "severity": "CRITICAL" if level_pct >= CRITICAL_THRESHOLD else "HIGH"
    }
    endpoints = ["/api/alert", "/api/alerts", "/api/notification", "/api/notify"]
    for ep in endpoints:
        try:
            r = requests.post(host.rstrip('/') + ep, json=payload, timeout=6)
            if r.status_code in (200,201):
                print(f"[ALERT] sent to {ep} for {node['node_id']} -> {payload['fill_level_pct']}%")
                return True
            else:
                # continue trying other endpoints
                print(f"[ALERT] {ep} responded {r.status_code}")
        except Exception as e:
            # ignore and try next endpoint
            # print minimal message to avoid noise
            print(f"[ALERT] failed to {ep}: {e}")
    # if we get here, no endpoint accepted it
    print(f"[ALERT] No alert endpoint accepted the message for {node['node_id']}.")
    return False

def assign_truck(host, fleet: TruckFleet, node):
    # find nearest idle truck
    nearest = fleet.find_nearest_idle(node["lat"], node["lon"])
    if nearest is None:
        print(f"[ASSIGN] No idle trucks available for {node['node_id']}. Will try fallback strategy.")
        # fallback: pick nearest truck even if assigned (least-bad choice)
        snapshot = fleet.get_snapshot()
        if not snapshot:
            print("[ASSIGN] No trucks in fleet at all.")
            return False
        nearest = min(snapshot, key=lambda t: haversine_km(node["lat"], node["lon"], t["lat"], t["lon"]))
        print(f"[ASSIGN] Fallback selected {nearest['truck_id']} (status={nearest['status']}).")

    distance_km = haversine_km(node["lat"], node["lon"], nearest["lat"], nearest["lon"])
    # estimate ETA in minutes (distance / speed) converted to minutes
    eta_minutes = (distance_km / TRUCK_SPEED_KMPH) * 60.0 if TRUCK_SPEED_KMPH > 0 else None

    assign_payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "truck_id": nearest["truck_id"],
        "node_id": node["node_id"],
        "node_lat": round(node["lat"],6),
        "node_lon": round(node["lon"],6),
        "truck_lat": round(nearest["lat"],6),
        "truck_lon": round(nearest["lon"],6),
        "distance_km": round(distance_km,3),
        "eta_minutes": round(eta_minutes,1) if eta_minutes is not None else None,
        "source": "simulator"
    }

    # try several endpoints for compatibility
    endpoints = ["/api/assign_truck", "/api/dispatch", "/api/assign"]
    success = False
    for ep in endpoints:
        try_count = 0
        while try_count <= ASSIGNMENT_RETRY:
            try:
                r = requests.post(host.rstrip('/') + ep, json=assign_payload, timeout=6)
                if r.status_code in (200,201):
                    print(f"[ASSIGN] Assigned {nearest['truck_id']} -> {node['node_id']} via {ep} (ETA {assign_payload['eta_minutes']}m).")
                    success = True
                    break
                else:
                    print(f"[ASSIGN] {ep} responded {r.status_code}: {r.text}")
            except Exception as e:
                print(f"[ASSIGN] Failed to post to {ep} (attempt {try_count+1}): {e}")
            try_count += 1
        if success:
            break

    # If assignment accepted by backend (or we choose to consider it assigned locally even if backend isn't reachable),
    # mark truck assigned locally and schedule release.
    # We'll mark it assigned even if network failed, to avoid repeated immediate reassign attempts.
    # If you prefer not to, change this behavior.
    fleet.mark_assigned(nearest["truck_id"], node["node_id"])
    fleet.release_after(nearest["truck_id"], SERVICE_DURATION)
    if not success:
        print(f"[ASSIGN] Assignment POSTs failed for {nearest['truck_id']} -> {node['node_id']}. Marked locally as assigned anyway.")
    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between sends")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--trucks", type=int, default=4)
    args = parser.parse_args()

    print(f"[SIM] Sending to {args.host}/api/sensor every {args.interval}s | nodes={args.nodes} | trucks={args.trucks}")
    nodes = make_nodes(args.nodes)
    fleet = TruckFleet(make_trucks(args.trucks))

    try:
        while True:
            for node in nodes:
                node = step_node(node)
                payload = post_sensor(args.host, node)

                # Overflow / high fill check & alert/assignment
                fill = payload["fill_level_pct"]
                if fill >= OVERFLOW_THRESHOLD:
                    print(f"[CHECK] {node['node_id']} >= {OVERFLOW_THRESHOLD}% -> trigger alert & assignment.")
                    # send alert (best-effort)
                    send_alert(args.host, node, fill)
                    # attempt assignment (best-effort)
                    assign_truck(args.host, fleet, node)
                elif fill >= (OVERFLOW_THRESHOLD - 10):
                    # optional pre-alert for near-overflow
                    print(f"[CHECK] {node['node_id']} nearing overflow ({fill}%). Consider pre-emptive action.")
                time.sleep(0.15)  # slight spread between nodes
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped by user")

if __name__ == "__main__":
    main()
