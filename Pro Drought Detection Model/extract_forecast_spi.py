# extract_forecast_spi.py

import os
import numpy as np
import xarray as xr
from tqdm import tqdm

# Settings
RAW_DATA_DIR = "data/raw"
FORECAST_PATCH_DIR = "data/spi_forecast_sequences"
PATCH_SIZE = 32
STRIDE = 16
INPUT_WINDOW = 10
FORECAST_OFFSET = 1  # Forecast SPI at T+1

os.makedirs(FORECAST_PATCH_DIR, exist_ok=True)

variables = ["monthly_rain", "max_temp", "min_temp", "radiation", "spi_1"]

# Forecast SPI (regression target)
def extract_forecast_spi_patches(region_file):
    ds = xr.open_dataset(os.path.join(RAW_DATA_DIR, region_file))
    region_name = region_file.split("_Combined_Consecutive.nc")[0]
    region_dir = os.path.join(FORECAST_PATCH_DIR, region_name)
    os.makedirs(region_dir, exist_ok=True)

    lat_dim, lon_dim, time_dim = ds.dims['lat'], ds.dims['lon'], ds.dims['time']

    for lat in range(0, lat_dim - PATCH_SIZE + 1, STRIDE):
        for lon in range(0, lon_dim - PATCH_SIZE + 1, STRIDE):
            for t in range(0, time_dim - INPUT_WINDOW - FORECAST_OFFSET):

                # Extract past input window (T-9 to T)
                patch_seq = []
                for var in variables:
                    data_slice = ds[var].isel(
                        time=slice(t, t + INPUT_WINDOW),
                        lat=slice(lat, lat + PATCH_SIZE),
                        lon=slice(lon, lon + PATCH_SIZE)
                    ).values
                    patch_seq.append(data_slice)

                patch_seq = np.stack(patch_seq, axis=-1)  # shape: (T, H, W, C)
                patch_seq = np.nan_to_num(patch_seq, nan=0.0)

                # Forecast label: SPI at T+1
                spi_future = ds['spi_1'].isel(
                    time=t + INPUT_WINDOW,
                    lat=slice(lat, lat + PATCH_SIZE),
                    lon=slice(lon, lon + PATCH_SIZE)
                ).values

                if np.isnan(spi_future).all():
                    continue  # skip fully NaN patches

                outname = f"{region_name}_lat{lat}_lon{lon}_t{t}"
                np.save(os.path.join(region_dir, outname + ".npy"), patch_seq)
                np.save(os.path.join(region_dir, outname + "_spi_label.npy"), spi_future)

    print(f"✅ Extracted SPI forecast data for region: {region_name}")

if __name__ == "__main__":
    for file in tqdm(os.listdir(RAW_DATA_DIR)):
        if file.endswith("_Combined_Consecutive.nc"):
            extract_forecast_spi_patches(file)
