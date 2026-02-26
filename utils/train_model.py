import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

# -------------------------
# Data Preparation
# -------------------------
train_dir = r"C:\Users\rachi\WasteProject\dataset\TRAIN"
val_dir   = r"C:\Users\rachi\WasteProject\dataset\TEST"


img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# -------------------------
# Model Architecture
# -------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------
# Training
# -------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------
# Save model and class labels
# -------------------------
model.save("waste_classifier.h5")

with open("class_labels.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Model and class labels saved!")
