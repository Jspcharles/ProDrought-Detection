# train_model_a5.py

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cnn_lstm_models import build_model

# Configs
MODEL_TYPE = "A5"
SEQUENCE_DIR = r"data/sequences"
MODEL_SAVE_PATH = rf"models/best_model_{MODEL_TYPE}.h5"
HISTORY_SAVE_PATH = rf"models/history_{MODEL_TYPE}.pkl"

# Load sequence data
X, Y = [], []
for region in os.listdir(SEQUENCE_DIR):
    region_path = os.path.join(SEQUENCE_DIR, region)
    for file in os.listdir(region_path):
        if file.endswith(".npy") and not file.endswith("_label.npy"):
            label_file = file.replace(".npy", "_label.npy")
            try:
                seq = np.load(os.path.join(region_path, file))
                label = np.load(os.path.join(region_path, label_file))
                if seq.shape == (10, 32, 32, 10):
                    X.append(seq)
                    Y.append(label)
            except:
                continue

X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
X, Y = shuffle(X, Y, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build model
model = build_model(MODEL_TYPE)
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
]

# Train model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=30,
    class_weight={0: 1.0, 1: 5.0},
    callbacks=callbacks,
    verbose=1
)

# Save training history
with open(HISTORY_SAVE_PATH, "wb") as f:
    pickle.dump(history.history, f)

print(f"\n✅ Training complete. Model saved to {MODEL_SAVE_PATH}")
