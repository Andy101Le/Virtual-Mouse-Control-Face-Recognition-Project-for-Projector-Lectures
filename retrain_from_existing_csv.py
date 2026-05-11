"""
retrain_from_existing_csv.py
────────────────────────────
Run this ONCE to fix your existing landmark_dataset.csv without
re-collecting any data.

Your CSV was saved with RAW screen coordinates, but main.py normalizes
at inference (wrist-relative + scale). This script:
  1. Loads the existing CSV
  2. Applies the same normalization main.py uses
  3. Retrains the model on the corrected data
  4. Saves the new model as landmark_gesture_model.h5 (overwrites old one)

Expected result: ~95% validation accuracy (up from ~20-25% before).

Usage:
    python retrain_from_existing_csv.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ── Config ────────────────────────────────────────────────────────────────────
GESTURES       = ['MOVE', 'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT']
DATASET_FILE   = 'landmark_dataset.csv'
MODEL_SAVE     = 'landmark_gesture_model.h5'
EPOCHS         = 300
BATCH_SIZE     = 32

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {DATASET_FILE} ...")
df    = pd.read_csv(DATASET_FILE)
X_raw = df.iloc[:, 1:].values   # 63 raw landmark coords
y     = df['label'].values.astype(int)

print(f"  {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
print("  Class distribution:")
for i, name in enumerate(GESTURES):
    print(f"    {i} {name}: {int(np.sum(y == i))} samples")

# ── Normalize — same logic as main.py ─────────────────────────────────────────
#   1. Subtract wrist (landmark 0) → position-invariant
#   2. Divide by max absolute value → scale-invariant
def normalize_row(row):
    pts     = row.reshape(21, 3)
    wrist   = pts[0]
    centered = pts - wrist
    scale   = np.max(np.abs(centered)) or 1.0
    return (centered / scale).flatten()

print("\nNormalizing landmarks (wrist-relative + scale) ...")
X = np.array([normalize_row(r) for r in X_raw], dtype=np.float32)
print(f"  Done. Feature range: [{X.min():.3f}, {X.max():.3f}]  (should be approx [-1, 1])")

# ── Split ─────────────────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_tr)}  |  Val: {len(X_val)}")

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
    EarlyStopping(
        monitor='val_accuracy', patience=25,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, verbose=1
    )
]

print("\nTraining ...")
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
print("\nPer-class report:")
print(classification_report(y_val, y_pred, target_names=GESTURES))

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE)
print(f"\nModel saved → {MODEL_SAVE}")
print("You can now run main.py — no other changes needed.")
