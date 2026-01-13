import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

region = "CHC"
DATA_DIR = f"data/prediction/{region}"

meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
pred = pd.read_csv(os.path.join(DATA_DIR, "predictions_2008.csv"))

df = meta.merge(pred, on="filename")

# assume original grid size known
LAT = 128   # change to ds.sizes['lat']
LON = 128   # change to ds.sizes['lon']
grid_stride = 16
patch_size = 32

for month in sorted(df["month"].unique()):
    month_df = df[df["month"] == month]

    grid = np.zeros((LAT, LON))
    grid[:] = np.nan

    for _, row in month_df.iterrows():
        lat = int(row.lat)
        lon = int(row.lon)
        label = row.predicted_label

        grid[lat:lat+patch_size, lon:lon+patch_size] = label

    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap="Purples", vmin=0, vmax=1)
    plt.title(f"{region} — Predicted Protracted Drought — {month}/2008")
    plt.colorbar(label="Protracted Drought (1=yes)")
    plt.savefig(os.path.join(DATA_DIR, f"map_2008_{month}.png"))
    plt.close()

print("✅ Maps saved for each month.")
