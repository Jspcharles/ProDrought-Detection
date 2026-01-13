import os
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ==============================
# Configuration
# ==============================
MODEL_PATH = "models/best_model_A8.h5"
DATA_DIR = r"data/prediction/all_years/CHC"
OUTPUT_CSV = os.path.join(DATA_DIR, "predictions_timeseries_2000_2024.csv")

BATCH_SIZE = 32
THRESHOLD = 0.5

# ==============================
# Load model
# ==============================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ==============================
# Load metadata
# ==============================
meta_df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
meta_df["filepath"] = meta_df["filename"].apply(
    lambda x: os.path.join(DATA_DIR, x)
)
meta_df = meta_df[meta_df["filepath"].apply(os.path.exists)]
meta_df = meta_df.reset_index(drop=True)

print(f"✅ Total patches to predict: {len(meta_df)}")

# ==============================
# Batched prediction
# ==============================
results = []

num_samples = len(meta_df)

for start in range(0, num_samples, BATCH_SIZE):
    end = min(start + BATCH_SIZE, num_samples)
    batch_df = meta_df.iloc[start:end]

    # Load batch
    batch_patches = []
    for fp in batch_df["filepath"]:
        batch_patches.append(np.load(fp))

    batch_patches = np.stack(batch_patches, axis=0)

    # Predict batch
    probs = model.predict(batch_patches, verbose=0).reshape(-1)

    labels = (probs >= THRESHOLD).astype(int)

    # Collect results
    for i, row in batch_df.iterrows():
        idx = i - start
        results.append([
            row["filename"],
            row["lat_idx"],
            row["lon_idx"],
            row["year"],
            row["month"],
            float(probs[idx]),
            int(labels[idx])
        ])

    if start % (BATCH_SIZE * 50) == 0:
        print(f"  ⏳ Processed {end}/{num_samples} patches")

# ==============================
# Save CSV
# ==============================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "lat_idx",
        "lon_idx",
        "year",
        "month",
        "probability",
        "predicted_label"
    ])
    writer.writerows(results)

print("✅ Predictions completed")
print("📄 Saved to:", OUTPUT_CSV)
