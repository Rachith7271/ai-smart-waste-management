# train_advanced.py
"""
Updated advanced training script (complete, ready-to-run).

Features:
- EfficientNetV2B0 backbone (ImageNet pretrained)
- TF-native MixUp (works inside tf.data pipeline)
- Simple on-the-fly augmentation (flip + brightness)
- Label smoothing option
- Cosine decay LR schedule
- Class weights computed from training folder
- Two-stage training: head (frozen) -> fine-tune (unfreeze top)
- Saves final model (.h5) and class labels (index -> class_name JSON)

Usage:
    - Edit TRAIN_DIR / VAL_DIR paths below if needed
    - pip install -r requirements (tensorflow, numpy, pillow)
    - python train_advanced.py
"""

import os
import json
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input

# ========== USER CONFIG ==========
TRAIN_DIR = r"C:\Users\rachi\WasteProject\dataset\TRAIN"
VAL_DIR   = r"C:\Users\rachi\WasteProject\dataset\TEST"
IMG_SIZE = (224, 224)
BATCH_SIZE = 24               # lower if you run out of memory
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 20
MODEL_OUT = "waste_classifier_effnetv2_adv.h5"
LABELS_OUT = "class_labels.json"
BEST_WEIGHTS = "best_effnetv2_adv.h5"

SEED = 42
MIXUP_ALPHA = 0.2            # set 0.0 to disable mixup
LABEL_SMOOTHING = 0.1       # set 0.0 to disable
USE_AUGMENT = True
AUTOTUNE = tf.data.AUTOTUNE
# ==================================

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------- helper: count image files in directory ----------
def count_images_in_dir(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count

train_count = count_images_in_dir(TRAIN_DIR)
val_count = count_images_in_dir(VAL_DIR)
if train_count == 0:
    raise ValueError(f"No training images found under {TRAIN_DIR}")
print(f"Found {train_count} train images and {val_count} val images.")

# ---------- Build tf.data datasets using image_dataset_from_directory ----------
# This returns integer labels and class_names in alphabetical order of folders.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names, " (num_classes=", num_classes, ")")

# Save labels mapping index -> class_name (string keys)
index_to_class = {str(i): name for i, name in enumerate(class_names)}
with open(LABELS_OUT, "w") as f:
    json.dump(index_to_class, f, indent=2)
print("Saved label mapping to", LABELS_OUT)

# ---------- Compute class weights ----------
# gather integer labels from directory walk via image_dataset_from_directory
# We can build counts by iterating once over train_ds (cheap)
label_list = []
for _, labels in train_ds.unbatch():
    # labels is scalar tensor
    label_list.append(int(labels.numpy()))
label_array = np.array(label_list)
unique, counts = np.unique(label_array, return_counts=True)
class_counts = dict(zip(unique.tolist(), counts.tolist()))
total = label_array.shape[0]
# protect against classes missing
for i in range(num_classes):
    class_counts.setdefault(i, 1)
class_weights = {int(k): float(total / (num_classes * v)) for k, v in class_counts.items()}
print("Class counts:", class_counts)
print("Class weights:", class_weights)

# ---------- TF-native MixUp ----------
def mixup_tf(images, labels, alpha):
    """TF-native MixUp applied to a batch.
    images: [B,H,W,C], labels: [B,num_classes] (one-hot)
    """
    if alpha <= 0.0:
        return images, labels

    batch_size = tf.shape(images)[0]
    # sample Beta via two Gamma draws
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = gamma1 / (gamma1 + gamma2)                          # shape [B]
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)

    mixed_images = images * tf.cast(lam_x, images.dtype) + images2 * tf.cast(1.0 - lam_x, images.dtype)
    mixed_labels = labels * tf.cast(lam_y, labels.dtype) + labels2 * tf.cast(1.0 - lam_y, labels.dtype)

    return mixed_images, mixed_labels

# ---------- Preprocessing functions ----------
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnetv2_preprocess

def prepare_for_training(images, labels):
    """Convert labels to one-hot, apply preprocess_input and optional augment+mixup (TF ops only)."""
    images = tf.cast(images, tf.float32)
    images = effnetv2_preprocess(images)  # model expects this preprocessing
    labels = tf.one_hot(labels, depth=num_classes)

    if USE_AUGMENT:
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, max_delta=0.06)
    if MIXUP_ALPHA > 0.0:
        images, labels = mixup_tf(images, labels, MIXUP_ALPHA)
    return images, labels

def prepare_for_validation(images, labels):
    images = tf.cast(images, tf.float32)
    images = effnetv2_preprocess(images)
    labels = tf.one_hot(labels, depth=num_classes)
    return images, labels

train_ds = train_ds.map(prepare_for_training, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(prepare_for_validation, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------- Build model ----------
base = EfficientNetV2B0(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet', pooling='avg')
base.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='swish')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs, name="effnetv2_adv_waste")
model.summary()

# ---------- Loss, optimizer, LR schedule ----------
if LABEL_SMOOTHING and LABEL_SMOOTHING > 0.0:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
else:
    loss_fn = 'categorical_crossentropy'

steps_per_epoch = max(1, train_count // BATCH_SIZE)
total_steps = (HEAD_EPOCHS + FINE_TUNE_EPOCHS) * steps_per_epoch
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=total_steps)

optimizer = optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# ---------- Callbacks ----------
checkpoint_cb = callbacks.ModelCheckpoint(BEST_WEIGHTS, save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
tensorboard_cb = callbacks.TensorBoard(log_dir='logs/effnetv2_adv', histogram_freq=1)

# ---------- Initial head training ----------
print("\n=== Head training (backbone frozen) ===\n")
history_head = model.fit(
    train_ds,
    epochs=HEAD_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=max(1, val_count // BATCH_SIZE),
    class_weight=class_weights,
    callbacks=[checkpoint_cb, reduce_lr_cb, earlystop_cb, tensorboard_cb]
)

# ---------- Fine-tuning ----------
print("\n=== Fine-tuning: unfreeze top layers of backbone ===\n")
base.trainable = True
# unfreeze last ~30-40% of base layers to avoid catastrophic forgetting
fine_tune_at = int(len(base.layers) * 0.7)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base.layers[fine_tune_at:]:
    layer.trainable = True

# recompile with lower learning rate
optimizer_fine = optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_fine, loss=loss_fn, metrics=['accuracy'])

history_fine = model.fit(
    train_ds,
    epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_head.epoch[-1] + 1 if history_head.epoch else 0,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=max(1, val_count // BATCH_SIZE),
    class_weight=class_weights,
    callbacks=[checkpoint_cb, reduce_lr_cb, earlystop_cb, tensorboard_cb]
)

# ---------- Load best weights if saved and save final model ----------
if os.path.exists(BEST_WEIGHTS):
    try:
        model.load_weights(BEST_WEIGHTS)
        print("Loaded best weights from", BEST_WEIGHTS)
    except Exception as e:
        print("Could not load best weights:", e)

model.save(MODEL_OUT)
print("✅ Saved model to", MODEL_OUT)

# ensure labels saved (already saved above, but rewrite to be safe)
index_to_class = {str(i): name for i, name in enumerate(class_names)}
with open(LABELS_OUT, "w") as f:
    json.dump(index_to_class, f, indent=2)
print("✅ Saved labels to", LABELS_OUT)

# ---------- Save merged history ----------
def merge_hist(h1, h2):
    merged = {}
    for k, v in h1.history.items():
        merged[k] = v.copy()
    for k, v in h2.history.items():
        merged.setdefault(k, []).extend(v)
    return merged

merged_history = merge_hist(history_head, history_fine)
with open("training_history_effnetv2_adv.json", "w") as f:
    json.dump(merged_history, f, indent=2)
print("✅ Saved training history to training_history_effnetv2_adv.json")

print("Training complete.")
