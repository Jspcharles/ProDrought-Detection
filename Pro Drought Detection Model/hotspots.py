import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

SEQUENCE_DIR = "data/sequences"
patch_size = 32
grid_shape = (256, 256)  # Adjust to match your full region size

drought_counts = defaultdict(lambda: np.zeros(grid_shape, dtype=int))

for region in os.listdir(SEQUENCE_DIR):
    region_path = os.path.join(SEQUENCE_DIR, region)
    for file in os.listdir(region_path):
        if file.endswith("_label.npy"):
            label_path = os.path.join(region_path, file)
            label = np.load(label_path).item()

            if label == 1:
                try:
                    parts = file.replace("_label.npy", "").split("_")
                    lat = int(parts[-3].replace("lat", ""))
                    lon = int(parts[-2].replace("lon", ""))
                    drought_counts[region][lat:lat+patch_size, lon:lon+patch_size] += 1
                except Exception as e:
                    print(f"Skipping {file}: {e}")

# Plot heatmaps
for region, heatmap in drought_counts.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, cmap="Reds", cbar_kws={'label': 'Drought Occurrence Count'})
    plt.title(f"Consecutive Drought Hotspots in {region}")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.tight_layout()
    plt.savefig(f"models/hotspots/drought_hotspots_{region}.png")
    plt.close()
