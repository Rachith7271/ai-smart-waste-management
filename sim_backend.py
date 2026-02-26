# sim_backend.py
from flask import Flask, request, jsonify, send_from_directory
import time
import threading

app = Flask(__name__, static_folder='static')

# In-memory store for assignments (for demo only)
assignments = {}  # truck_id -> assignment dict

assign_lock = threading.Lock()

@app.route('/api/sensor', methods=['POST'])
def api_sensor():
    data = request.get_json(force=True)
    print("[SENSOR] Received:", data.get("node_id"), data.get("fill_level_pct"))
    return jsonify({"status":"ok"}), 201

@app.route('/api/assign_truck', methods=['POST'])
def api_assign_truck():
    data = request.get_json(force=True)
    truck_id = data.get("truck_id")
    node_id = data.get("node_id")
    ts = time.time()
    if not truck_id or not node_id:
        return jsonify({"error":"truck_id and node_id required"}), 400

    with assign_lock:
        assignments[truck_id] = {
            "truck_id": truck_id,
            "node_id": node_id,
            "truck_lat": data.get("truck_lat"),
            "truck_lon": data.get("truck_lon"),
            "node_lat": data.get("node_lat"),
            "node_lon": data.get("node_lon"),
            "distance_km": data.get("distance_km"),
            "eta_minutes": data.get("eta_minutes"),
            "assigned_at": ts,
            "source": data.get("source", "simulator")
        }
    print(f"[ASSIGN RECEIVED] {truck_id} -> {node_id} (ETA {data.get('eta_minutes')}m)")
    return jsonify({"status":"assigned"}), 201

@app.route('/api/dispatch', methods=['POST'])
def api_dispatch_alias():
    # Accept dispatch as alias (simulator may try different endpoints)
    return api_assign_truck()

@app.route('/api/assignments', methods=['GET'])
def get_assignments():
    # return current assignments as list
    with assign_lock:
        resp = list(assignments.values())
    return jsonify(resp)

@app.route('/api/complete_assignment', methods=['POST'])
def complete_assignment():
    data = request.get_json(force=True)
    truck_id = data.get("truck_id")
    if not truck_id:
        return jsonify({"error":"truck_id required"}), 400
    with assign_lock:
        if truck_id in assignments:
            assignments.pop(truck_id)
            print(f"[ASSIGN COMPLETE] {truck_id} marked complete.")
            return jsonify({"status":"completed"}), 200
        else:
            return jsonify({"error":"not found"}), 404

# Optional: serve the static dashboard
@app.route('/')
def index():
    return send_from_directory('static','index.html')

if __name__ == '__main__':
    print("Starting sim backend on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
