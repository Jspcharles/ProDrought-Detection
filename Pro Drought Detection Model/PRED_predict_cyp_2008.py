import os
import csv
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/best_model_A8.h5"
DATA_DIR = "data/prediction/2002/CHC"

model = load_model(MODEL_PATH)

patch_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy") and not f.endswith("_label.npy")]

pred_csv = os.path.join(DATA_DIR, "predictions_2002.csv")
with open(pred_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "probability", "predicted_label"])

    for pf in patch_files:
        patch = np.load(os.path.join(DATA_DIR, pf))
        patch = np.expand_dims(patch, 0)

        prob = model.predict(patch, verbose=0)[0][0]
        label = 1 if prob >= 0.5 else 0
        writer.writerow([pf, prob, label])

print("✅ Predictions saved to", pred_csv)
