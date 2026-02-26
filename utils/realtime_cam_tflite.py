# realtime_cam_tflite.py
"""
Realtime webcam classification (fast) using TFLite.
- Uses moving average smoothing, confidence threshold, FPS display.
- Make sure you have a TFLite model (fp16 or fp32). Set TFLITE_MODEL path.
"""
import cv2, time, json, os, numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_MODEL = "waste_model_fp16.tflite"   # choose file produced by export_models.py
LABELS = "class_labels.json"
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.4
SMOOTHING = 6

if not os.path.exists(TFLITE_MODEL):
    raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL}. Run export_models.py first.")

with open(LABELS, "r") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

# Setup tflite interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
out_index = output_details[0]['index']

# camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (cv2.VideoCapture failed). Try different index.")

pred_buffer = []
prev_time = time.time()

def predict_frame_tflite(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype('float32'), axis=0)
    # If model was converted with fp16, interpreter likely expects float32/float16; we pass float32
    # If quantized (uint8) you must scale to uint8 using input_details info
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        arr = arr / 255.0  # assume model trained on [0,1]
        arr = arr / scale + zero_point
        arr = arr.astype(np.uint8)
    interpreter.set_tensor(input_index, arr)
    interpreter.invoke()
    out = interpreter.get_tensor(out_index)[0]
    # If output is quantized, dequantize
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        out = scale * (out.astype(np.float32) - zero_point)
    return out

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preds = predict_frame_tflite(frame)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    name = labels.get(idx, "Unknown")

    pred_buffer.append((name, conf))
    if len(pred_buffer) > SMOOTHING:
        pred_buffer.pop(0)

    # aggregate
    counts = {}
    confs = {}
    for n, c in pred_buffer:
        counts[n] = counts.get(n, 0) + 1
        confs[n] = confs.get(n, 0.0) + c
    best = max(counts.items(), key=lambda x: x[1])[0]
    avg_conf = confs[best] / counts[best]

    label_text = f"{best}: {avg_conf:.2f}" if avg_conf >= CONF_THRESHOLD else f"Unknown ({avg_conf:.2f})"

    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if cur_time != prev_time else 0.0
    prev_time = cur_time

    # overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 70), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Waste Classifier (TFLite) - q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
