import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

region_name = "CYP"

raw_file = r"data\Con Drought Labelled Files\CYP_Combined_Consecutive.nc"
prediction_dir = f"data/prediction/{region_name}"
metadata_csv = os.path.join(prediction_dir, "metadata.csv")
prediction_csv = os.path.join(prediction_dir, "predictions_2008.csv")

ds = xr.open_dataset(raw_file)
meta = pd.read_csv(metadata_csv)
pred = pd.read_csv(prediction_csv)

df = meta.merge(pred, on="filename")
df["month"] = df["month"].astype(int)

LAT = ds.sizes["lat"]
LON = ds.sizes["lon"]
PATCH_SIZE = 32
STRIDE = 16  # ✅ Ensure overlap

output_dir = os.path.join(prediction_dir, "pixelwise_maps_2008")
os.makedirs(output_dir, exist_ok=True)

unique_months = sorted(df["month"].unique())

for month in unique_months:
    print(f"📌 Pixel-wise fusion for Month: {month}/2008")

    # SPI base layer
    month_data = ds.sel(time=f"2008-{month:02d}").squeeze("time")
    spi = month_data["spi_1"].values
    spi_masked = np.ma.masked_where(np.isnan(spi), spi)

    # Probability & count grids
    prob_grid = np.zeros((LAT, LON), dtype=float)
    count_grid = np.zeros((LAT, LON), dtype=float)

    # Only patches for this month
    month_df = df[df["month"] == month]

    for _, row in month_df.iterrows():
        lat = int(row.lat)
        lon = int(row.lon)
        prob = float(row.probability)  # ✅ Use probability, not binary

        # Only update the valid region of the patch (in case of padding)
        h = min(PATCH_SIZE, LAT - lat)
        w = min(PATCH_SIZE, LON - lon)

        prob_grid[lat:lat+h, lon:lon+w] += prob
        count_grid[lat:lat+h, lon:lon+w] += 1

    # Average overlapping contributions
    count_grid[count_grid == 0] = 1
    pixel_prob = prob_grid / count_grid

    # Pixel-wise binary label
    pixel_label = (pixel_prob >= 0.5).astype(int)

    # Visualization
    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(f"Pixel-wise Predicted Protracted Drought — {month_data.time.dt.strftime('%B %Y').item()}", fontsize=14)

    spi_plot = ax.imshow(spi_masked, cmap="RdYlBu", vmin=-2, vmax=2, origin="lower")
    ax.contourf(pixel_label, levels=[0.5, 1], colors='blue', alpha=0.3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(spi_plot, ax=ax, orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("SPI (Standardized Precipitation Index)")

    out_name = os.path.join(output_dir, f"{region_name}_pixelwise_{month:02d}_2008.png")
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Saved:", out_name)

print("\n✅ Pixel-wise drought maps saved in:", output_dir)
