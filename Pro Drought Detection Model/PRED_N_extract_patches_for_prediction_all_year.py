import os
import numpy as np
import xarray as xr

# ==============================
# Configuration
# ==============================
RAW_DATA_DIR = r"data/raw"
PATCHES_DIR = r"data/prediction/all_years"
PATCH_SIZE = 32
STRIDE = 16
TIME_WINDOW = 10

region_name = "CHC"
region_file = f"{region_name}_Combined_Consecutive.nc"

variables = [
    "monthly_rain",
    "max_temp",
    "min_temp",
    "radiation",
    "spi_1"
]

START_YEAR = 2000
END_YEAR = 2024

os.makedirs(PATCHES_DIR, exist_ok=True)

# ==============================
# Patch Extraction Function
# ==============================
def extract_patches_all_years():

    # Load dataset
    ds = xr.open_dataset(os.path.join(RAW_DATA_DIR, region_file))

    # Select full analysis period
    ds = ds.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

    print(f"✅ Loaded {region_name} data: {START_YEAR}–{END_YEAR}")
    print(f"   Time steps: {ds.sizes['time']}")
    print(f"   Grid: {ds.sizes['lat']} x {ds.sizes['lon']}")

    # Output directory for this region
    region_dir = os.path.join(PATCHES_DIR, region_name)
    os.makedirs(region_dir, exist_ok=True)

    # Metadata file
    metadata_path = os.path.join(region_dir, "metadata.csv")
    with open(metadata_path, "w") as meta:
        meta.write("filename,lat_idx,lon_idx,year,month\n")

    patch_counter = 0

    # ==============================
    # Sliding window extraction
    # ==============================
    for lat in range(0, ds.sizes["lat"], STRIDE):
        for lon in range(0, ds.sizes["lon"], STRIDE):
            for t in range(0, ds.sizes["time"] - TIME_WINDOW + 1):

                patch_raw = []

                # Extract each variable
                for var in variables:
                    data_slice = ds[var].isel(
                        time=slice(t, t + TIME_WINDOW),
                        lat=slice(lat, lat + PATCH_SIZE),
                        lon=slice(lon, lon + PATCH_SIZE)
                    ).values

                    patch_raw.append(data_slice)

                # Stack variables → (T, H, W, C)
                patch_raw = np.stack(patch_raw, axis=-1)

                # Pad if near boundary
                patch = np.full(
                    (TIME_WINDOW, PATCH_SIZE, PATCH_SIZE, len(variables)),
                    np.nan,
                    dtype=np.float32
                )
                patch[:, :patch_raw.shape[1], :patch_raw.shape[2], :] = patch_raw

                # Create validity mask
                mask = (~np.isnan(patch)).astype(np.float32)

                # Replace NaNs with zeros
                patch = np.nan_to_num(patch, nan=0.0)

                # Concatenate data + mask
                final_patch = np.concatenate([patch, mask], axis=-1)

                # Time metadata (use last timestep in window)
                time_val = ds.time.values[t + TIME_WINDOW - 1]
                year = int(np.datetime_as_string(time_val, unit="Y"))
                month = int(np.datetime_as_string(time_val, unit="M")[-2:])

                # Save patch
                outname = f"{region_name}_lat{lat}_lon{lon}_t{t}"
                np.save(os.path.join(region_dir, outname + ".npy"), final_patch)

                # Write metadata
                with open(metadata_path, "a") as meta:
                    meta.write(
                        f"{outname}.npy,{lat},{lon},{year},{month}\n"
                    )

                patch_counter += 1

    print(f"✅ Patch extraction complete")
    print(f"   Total patches generated: {patch_counter}")

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    extract_patches_all_years()
