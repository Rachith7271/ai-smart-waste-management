# export_models.py  (robust, direct-from-Keras conversion)
import os, shutil, traceback
import tensorflow as tf
import numpy as np

KERAS_H5 = "best_effnetv2_adv.h5"   # change if different
TFLITE_FP32 = "waste_model_fp32.tflite"
TFLITE_FP16 = "waste_model_fp16.tflite"
TFLITE_INT8 = "waste_model_int8.tflite"
SAVED_MODEL_DIR = "saved_model_fallback"

if not os.path.exists(KERAS_H5):
    raise FileNotFoundError(f"Keras model not found: {KERAS_H5}")

print("Loading Keras model...")
model = tf.keras.models.load_model(KERAS_H5)
print("Model loaded. Attempting direct TFLite conversion from Keras model...")

def convert_from_keras(model):
    # FP32
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_fp32 = converter.convert()
        with open(TFLITE_FP32, "wb") as f:
            f.write(tflite_fp32)
        print("Saved TFLite FP32:", TFLITE_FP32)
    except Exception as e:
        print("FP32 conversion from Keras failed:", e)
        raise

    # FP16
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        with open(TFLITE_FP16, "wb") as f:
            f.write(tflite_fp16)
        print("Saved TFLite FP16:", TFLITE_FP16)
    except Exception as e:
        print("FP16 conversion from Keras failed:", e)
        raise

    # INT8 dynamic range (no representative dataset)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_int8 = converter.convert()
        with open(TFLITE_INT8, "wb") as f:
            f.write(tflite_int8)
        print("Saved TFLite INT8 (dynamic):", TFLITE_INT8)
    except Exception as e:
        print("INT8 conversion (dynamic) failed:", e)
        # don't re-raise — int8 is optional

# Try direct conversion first
try:
    convert_from_keras(model)
    print("Direct conversion from Keras -> TFLite succeeded.")
except Exception:
    print("Direct conversion failed — falling back to SavedModel export using a concrete function.")
    # If direct conversion fails, fall back to SavedModel via concrete function (as last resort)
    try:
        # create concrete function
        try:
            input_shape = model.inputs[0].shape.as_list()
            _, H, W, C = input_shape
            if H is None or W is None:
                H, W, C = 224, 224, 3
        except Exception:
            H, W, C = 224, 224, 3

        @tf.function(input_signature=[tf.TensorSpec([None, H, W, C], tf.float32, name="serving_input")])
        def serve_fn(inputs):
            return {"outputs": model(inputs, training=False)}

        concrete_fn = serve_fn.get_concrete_function()

        if os.path.exists(SAVED_MODEL_DIR):
            shutil.rmtree(SAVED_MODEL_DIR)
        tf.saved_model.save(model, SAVED_MODEL_DIR, signatures=concrete_fn)
        print("Saved fallback SavedModel to", SAVED_MODEL_DIR)

        # Convert using saved model
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        tflite_fp32 = converter.convert()
        with open(TFLITE_FP32, "wb") as f:
            f.write(tflite_fp32)
        print("Saved TFLite FP32 (via SavedModel):", TFLITE_FP32)

        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        with open(TFLITE_FP16, "wb") as f:
            f.write(tflite_fp16)
        print("Saved TFLite FP16 (via SavedModel):", TFLITE_FP16)

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_int8 = converter.convert()
            with open(TFLITE_INT8, "wb") as f:
                f.write(tflite_int8)
            print("Saved TFLite INT8 (via SavedModel):", TFLITE_INT8)
        except Exception as e:
            print("INT8 conversion via SavedModel failed:", e)

    except Exception as fallback_err:
        print("Fallback SavedModel export failed. Full traceback below:")
        traceback.print_exc()
        raise fallback_err

print("Export process finished. Check for tflite files in the working directory.")
