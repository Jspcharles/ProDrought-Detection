import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# === Global Parameters ===
YEAR = 2002
region_name = "CHC"

# === Files & Paths ===
raw_file = r"data\Con Drought Labelled Files\CHC_Combined_Consecutive.nc"

prediction_dir = f"data/prediction/{YEAR}/{region_name}"
metadata_csv = os.path.join(prediction_dir, "metadata.csv")
prediction_csv = os.path.join(prediction_dir, f"predictions_{YEAR}.csv")

# === Load original dataset (for SPI background) ===
ds = xr.open_dataset(raw_file)
print("✅ Loaded region file:", raw_file)

# === Load metadata + predictions ===
meta = pd.read_csv(metadata_csv)
pred = pd.read_csv(prediction_csv)
df = meta.merge(pred, on="filename")

# Convert month from string to int (safety)
df["month"] = df["month"].astype(int)

# === Region grid & patch size ===
LAT = ds.sizes["lat"]
LON = ds.sizes["lon"]
PATCH_SIZE = 32

# === Output folder ===
output_dir = os.path.join(prediction_dir, f"maps_{YEAR}")
os.makedirs(output_dir, exist_ok=True)

unique_months = sorted(df["month"].unique())

for month in unique_months:
    print(f"📌 Processing Month: {month}/{YEAR}")

    # === Select monthly SPI map ===
    month_data = ds.sel(time=f"{YEAR}-{month:02d}").squeeze("time")
    spi = month_data["spi_1"].values
    spi_masked = np.ma.masked_where(np.isnan(spi), spi)

    # === Build empty drought mask ===
    drought_mask = np.full((LAT, LON), np.nan)

    # Get predictions only for this month
    month_df = df[df["month"] == month]

    # Fill mask with predicted labels from each patch
    for _, row in month_df.iterrows():
        lat = int(row.lat)
        lon = int(row.lon)
        label = row.predicted_label
        drought_mask[lat:lat + PATCH_SIZE, lon:lon + PATCH_SIZE] = label

    # === Visualization ===
    fig, ax = plt.subplots(figsize=(6, 6))
    date_label = month_data.time.dt.strftime('%B %Y').item()

    fig.suptitle(
        f"Predicted Protracted Drought — {region_name} — {date_label}",
        fontsize=14
    )

    # SPI background
    spi_plot = ax.imshow(
        spi_masked,
        cmap="RdYlBu",
        vmin=-2,
        vmax=2,
        origin="lower"
    )

    # Overlay drought (blue)
    ax.contourf(
        drought_mask,
        levels=[0.5, 1],
        colors="blue",
        alpha=0.3
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(date_label)

    # Colorbar
    cbar = fig.colorbar(
        spi_plot,
        ax=ax,
        orientation="vertical",
        fraction=0.05,
        pad=0.02
    )
    cbar.set_label("SPI (Standardized Precipitation Index)")

    # Save
    out_name = os.path.join(
        output_dir,
        f"{region_name}_predicted_{month:02d}_{YEAR}.png"
    )
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Saved:", out_name)

print(f"\n✅ All predicted monthly maps saved in: {output_dir}")
