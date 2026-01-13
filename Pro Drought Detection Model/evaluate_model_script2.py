# quick_evaluate_model.py

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from data_generator import SequenceDataGenerator, get_region_dirs

MODEL_ID = "A8"
MODEL_PATH = f"models/best_model_{MODEL_ID}.h5"

# Load model
model = load_model(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# Load small sample from test data
all_dirs = get_region_dirs("data/sequences")
all_dirs.sort()
test_dirs = all_dirs[-6:]  # use only last 3 regions for speed

test_gen = SequenceDataGenerator(test_dirs, batch_size=16, shuffle=False)

# Run prediction on first 100 valid samples
y_true, y_pred = [], []
sample_count = 0

for i, (X_batch, y_batch) in enumerate(test_gen):
    if len(X_batch) == 0:
        continue

    preds = model.predict(X_batch, verbose=0)
    y_true.extend(y_batch)
    y_pred.extend(preds.flatten())
    sample_count += len(y_batch)

    print(f"✅ Batch {i+1}: {len(y_batch)} samples")

    if sample_count >= 300:
        break

if not y_true:
    print("⚠️ No valid samples found in test subset.")
    exit()

# Threshold and report
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_binary, digits=4))
print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))
print(f"\n🔥 ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
