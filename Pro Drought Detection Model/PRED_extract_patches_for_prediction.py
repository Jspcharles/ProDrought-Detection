import os
import numpy as np
import xarray as xr

RAW_DATA_DIR = r"data/raw"
PATCHES_DIR = r"data/prediction/2002"
PATCH_SIZE = 32
STRIDE = 16
TIME_WINDOW = 10

region_name = "CHC"
region_file = f"{region_name}_Combined_Consecutive.nc"

os.makedirs(PATCHES_DIR, exist_ok=True)

def extract_patches_for_year():
    ds = xr.open_dataset(os.path.join(RAW_DATA_DIR, region_file))
    
    # Select only year 2008
    ds_2008 = ds.sel(time=slice("2001-01-01", "2002-12-31"))

    region_dir = os.path.join(PATCHES_DIR, region_name)
    os.makedirs(region_dir, exist_ok=True)

    metadata_path = os.path.join(region_dir, "metadata.csv")
    with open(metadata_path, "w") as meta:
        meta.write("filename,lat,lon,month\n")

    for lat in range(0, ds_2008.sizes['lat'], STRIDE):
        for lon in range(0, ds_2008.sizes['lon'], STRIDE):
            for t in range(0, ds_2008.sizes['time'] - TIME_WINDOW + 1):
                
                patch_raw = []
                for var in ["monthly_rain", "max_temp", "min_temp", "radiation", "spi_1"]:
                    data_slice = ds_2008[var].isel(
                        time=slice(t, t + TIME_WINDOW),
                        lat=slice(lat, lat + PATCH_SIZE),
                        lon=slice(lon, lon + PATCH_SIZE)
                    ).values
                    patch_raw.append(data_slice)

                patch_raw = np.stack(patch_raw, axis=-1)

                # make full patch (padding if needed)
                patch = np.full((TIME_WINDOW, PATCH_SIZE, PATCH_SIZE, 5), np.nan)
                patch[:, :patch_raw.shape[1], :patch_raw.shape[2], :] = patch_raw

                # create mask
                mask = (~np.isnan(patch)).astype(np.float32)
                patch = np.nan_to_num(patch, nan=0.0)

                final_patch = np.concatenate([patch, mask], axis=-1)

                outname = f"{region_name}_lat{lat}_lon{lon}_t{t}"
                np.save(os.path.join(region_dir, outname + ".npy"), final_patch)

                month_idx = ds_2008.time.values[t + TIME_WINDOW - 1]  # last timestep month
                with open(metadata_path, "a") as meta:
                    meta.write(f"{outname}.npy,{lat},{lon},{np.datetime_as_string(month_idx, unit='M')[-2:]}\n")

    print("✅ Patch extraction for 2008 complete.")

if __name__ == "__main__":
    extract_patches_for_year()
