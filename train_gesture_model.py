# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # ----------------------------
# # CONFIG
# # ----------------------------
# DATASET_DIR = "hand_dataset"
# IMAGE_SIZE = 128
# BATCH_SIZE = 32
# EPOCHS = 15
# MODEL_SAVE_PATH = "gesture_model.h5"
#
# # ----------------------------
# # Data generators
# # ----------------------------
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,  # 20% for validation
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True
# )
#
# train_gen = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     color_mode='rgb',
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )
#
# val_gen = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     color_mode='rgb',
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=False
# )
#
# # ----------------------------
# # CNN Model
# # ----------------------------
# num_classes = len(train_gen.class_indices)
#
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#     MaxPooling2D((2,2)),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D((2,2)),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D((2,2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
#
# # ----------------------------
# # Train the model
# # ----------------------------
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS
# )
#
# # ----------------------------
# # Save model
# # ----------------------------
# model.save(MODEL_SAVE_PATH)
# print(f"Model saved to {MODEL_SAVE_PATH}")
# print("Classes:", train_gen.class_indices)





# ##First WORKING VERSION##
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# import os
#
# # ========================= CONFIG =========================
# GESTURES = ["MOVE", "LEFT CLICK", "RIGHT CLICK", "ZOOM IN", "ZOOM OUT"]
# DATASET_FILE = "landmark_dataset.csv"
# MODEL_SAVE_PATH = "landmark_gesture_model.h5"
# EPOCHS = 2000
# BATCH_SIZE = 64
#
# # ========================= LOAD DATA =========================
# if not os.path.exists(DATASET_FILE):
#     raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found!")
#
# data = np.loadtxt(DATASET_FILE, delimiter=",", skiprows=1)  # skip header if present
#
# X = data[:, 1:]   # 63 landmark features (x, y, z for 21 points)
# y = data[:, 0].astype(int)  # gesture labels (0 to N-1)
#
# print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
# print(f"Classes: {GESTURES}")
#
# # Optional: One-hot encode if you prefer categorical crossentropy
# # y = to_categorical(y, num_classes=len(GESTURES))
#
# # ========================= BUILD MODEL =========================
# model = Sequential([
#     Dense(256, activation='relu', input_shape=(63,)),
#     Dropout(0.4),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(len(GESTURES), activation='softmax')
# ])
#
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',   # use this when y is integer labels
#     # loss='categorical_crossentropy',        # use this if you one-hot encoded y
#     metrics=['accuracy']
# )
#
# model.summary()
#
# # ========================= TRAIN =========================
# history = model.fit(
#     X, y,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     validation_split=0.2,      # 20% validation split
#     shuffle=True,
#     verbose=1
# )
#
# # ========================= SAVE MODEL =========================
# model.save(MODEL_SAVE_PATH)
# print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")
# print("Gesture classes:", dict(enumerate(GESTURES)))








"""
train_gesture_model.py
──────────────────────
Train the gesture MLP from a landmark_dataset.csv that was collected
with the NORMALIZED collect_gestures.py (wrist-relative + scale).

If you still have OLD raw data, run retrain_from_existing_csv.py instead
— it applies the normalization on the fly so you don't need to recollect.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os

# ── Config ────────────────────────────────────────────────────────────────────
GESTURES     = ['MOVE', 'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT']
DATASET_FILE = 'landmark_dataset.csv'
MODEL_SAVE   = 'landmark_gesture_model.h5'
EPOCHS       = 300
BATCH_SIZE   = 32

# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"'{DATASET_FILE}' not found. Run collect_gestures.py first.")

df = pd.read_csv(DATASET_FILE)
X  = df.iloc[:, 1:].values.astype(np.float32)
y  = df['label'].values.astype(int)

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
print("Class distribution:")
for i, name in enumerate(GESTURES):
    print(f"  {i} {name}: {int(np.sum(y == i))} samples")

# Sanity check: if wrist coords (first 3 features) vary widely, data is NOT normalized
wrist_x_range = X[:, 0].max() - X[:, 0].min()
if wrist_x_range > 0.05:
    print("\n⚠️  WARNING: Data does not appear to be normalized (wrist x ranges "
          f"{X[:,0].min():.3f}–{X[:,0].max():.3f}).")
    print("   Run retrain_from_existing_csv.py instead, or recollect with the new collect_gestures.py.\n")

# ── Split ─────────────────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,), kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.2),

    Dense(len(GESTURES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy : {val_acc:.2%}")
print(f"Validation loss     : {val_loss:.4f}")
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print(classification_report(y_val, y_pred, target_names=GESTURES))

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE)
print(f"Model saved → {MODEL_SAVE}")
