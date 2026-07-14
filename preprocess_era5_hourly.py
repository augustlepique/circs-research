
"""
Preprocess raw hourly ERA5 files for TempestExtremes tracking (1996-2025).

For each monthly file:
  - rename time dim  valid_time -> time   (TE expects 'time')
  - drop GRIB metadata coords 'number' and 'expver' (if present)
  - for z: convert geopotential (m^2/s^2) -> geopotential height (m), /g

Matches the validated 2020 pilot preprocessing.

Raw inputs expected at:
  {RAW_DIR}/era5_msl_{YEAR}_{MM}.nc
  {RAW_DIR}/era5_z_{YEAR}_{MM}.nc
Processed outputs written to:
  {OUT_DIR}/era5_msl_{YEAR}_{MM}_processed.nc
  {OUT_DIR}/era5_z_{YEAR}_{MM}_processed.nc

Resumable: skips a month/variable whose processed file already exists.
Hard-fails: exits non-zero if a raw input file is missing.

Edit RAW_DIR / OUT_DIR / filename patterns below to match your layout.
"""

import os
import sys
import xarray as xr

## User Settings
START_YEAR = 1995
END_YEAR = 2025

RAW_DIR = "/data1/lepique/era5_hourly_1996_2025"  # raw downloaded files
OUT_DIR = "/data1/lepique/TempestExtremes/era5_hourly_1996_2025_processed" # processed outputs

G = 9.80665  # to convert z to geopotential height
DROP_COORDS = ["number","expver"] # GRIB metadata to drop

os.makedirs(OUT_DIR, exist_ok=True)

def raw_path(var, year, mm):
    return os.path.join(RAW_DIR, f"era5_{var}_{year}_{mm}.nc")


def out_path(var, year, mm):
    return os.path.join(OUT_DIR, f"era5_{var}_{year}_{mm}_processed.nc")


def process_file(var, year, mm):
    src = raw_path(var, year, mm)
    dst = out_path(var, year, mm)

    # Resumable: skip if already done
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"SKIP {var} {year}-{mm} (processed file exists)")
        return

    # Hard-fail on missing input
    if not os.path.isfile(src):
        sys.exit(f"ERROR: missing raw input {src}")

    print(f"Processing {var} {year}-{mm} ...")
    ds = xr.open_dataset(src)

    # rename time dim if needed
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # drop GRIB metadata coords/vars if present
    drop = [c for c in DROP_COORDS if c in ds.variables]
    if drop:
        ds = ds.drop_vars(drop)

    # convert geopotential -> geopotential height for z
    if var == "z":
        if "z" not in ds.variables:
            ds.close()
            sys.exit(f"ERROR: variable 'z' not found in {src}; vars = {list(ds.data_vars)}")
        ds["z"] = ds["z"] / G
        ds["z"].attrs["units"] = "m"
        ds["z"].attrs["long_name"] = "geopotential height"
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(ds["longitude"] % 360)).sortby("longitude")


    # write
    tmp = dst + ".tmp"
    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, dst)   # atomic: a half-written file never looks "done"
    print(f"  -> {dst}")


def main():
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            mm = f"{month:02d}"
            for var in ("msl", "z"):
                process_file(var, year, mm)
    print("DONE.")


if __name__ == "__main__":
    main()

