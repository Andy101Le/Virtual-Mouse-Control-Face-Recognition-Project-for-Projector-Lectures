import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os

GESTURES     = ["MOVE", "LEFT CLICK", "RIGHT CLICK", "ZOOM IN", "ZOOM OUT"]
DATASET_FILE = "landmark_dataset.csv"
MODEL_SAVE   = "landmark_gesture_model.h5"
EPOCHS       = 300
BATCH_SIZE   = 32

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError("Run collect_gestures.py first to build %s" % DATASET_FILE)

df = pd.read_csv(DATASET_FILE)
X  = df.iloc[:, 1:].values.astype(np.float32)
y  = df["label"].values.astype(int)

print("Loaded %d samples, %d features" % (X.shape[0], X.shape[1]))
for i, name in enumerate(GESTURES):
    print("  %d %s: %d samples" % (i, name, int(np.sum(y == i))))

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential([
    Dense(256, activation="relu", input_shape=(63,), kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(len(GESTURES), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1),
]

model.fit(X_tr, y_tr,
          validation_data=(X_val, y_val),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=callbacks,
          verbose=1)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print("Validation accuracy: %.2f%%" % (val_acc * 100))
print("Validation loss:     %.4f"   % val_loss)

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print(classification_report(y_val, y_pred, target_names=GESTURES))

model.save(MODEL_SAVE)
print("Model saved -> %s" % MODEL_SAVE)
