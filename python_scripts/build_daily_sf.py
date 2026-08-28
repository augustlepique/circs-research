"""
build_daily_sf.py
=================
stage 1b: collapse hourly ERA5 'sf' into 12UTC centered daily accumulations

Daily field for day D = sum of the hourly sf over the window 12Z(D) -> 12Z(D+1),
which is centered on 0Z(D+1). We're doing this because NOHRSC snowfall data is natively
aggregated as 24h accumulations, and this also follows the Hawcroft et al. (2012)
published template. They do 0-0Z; instead we are doing 12-12Z, to follow SPC convective
day convention.

Writes one single-timestep file per day (daily_sf_YYYYMMDD.nc) because that is 
what the stage 2 NodeFileFilter template consumes, plus an --in_data_list per
season

Output conventions, matching preprocess_era5_hourly.py so TE sees one grid 
convention across the project:
    - longitude converted to 0...360 and sorted ascending
    - time dim named 'time' (instead of valid_time)
    - drop GRIB metadata (number, expver)
"""

#import glob
import sys
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr

IN_DIR = Path("/data1/lepique/era5_sf_hourly/")
OUT_DIR = Path("/data1/lepique/era5_sf_daily/")
DROP_COORDS = ["number", "expver"]

# ERA5 hourly accumulations are stamped at the end of their window: the value
# at valid_time t covers (t-1h, t). Shifting labels back one hour therefore
# restamps each sample by the start of its window, after which a plain daily
# resample sums exactly 12Z-12Z.
ACCUM_LABEL_IS_WINDOW_END = True

def open_sf(path):
    """
    Open an ERA5 sf file, tolerating the CDS zip-with-an-.nc-name case.
    
    A single variable request returns netCDF, but multi-variable can split by GRIB
    stepType into a zip (and are what my era5_sfcx_*.nc are). Handling it
    here so that this functions works against those files as well.
    """
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path)
    with zipfile.ZipFile(path) as zf:
        member = next((n for n in zf.namelist() if "accum" in n), None)
        if member is None:
            sys.exit(f"ERROR: {path} is a zip with no accum member: {zf.namelist()}")
        extracted = Path(path).with_suffix(".accum.nc")
        if not extracted.exists():
            extracted.write_bytes(zf.read(member))
    return xr.open_dataset(extracted)


def season_files(season, in_dir=IN_DIR):
    """Hourly files making up DJFM 'season', in chronological order, + Apr pad."""
    names = [f"era5_sf_{season - 1}_12.nc"] + \
            [f"era5_sf_{season}_{m:02d}.nc" for m in (1,2,3)]
    paths = [in_dir / n for n in names]
    missing = [ p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"ERROR: missing hourly input(s) {[p.name for p in missing]}")
    pad = in_dir / f"era5_sf_{season}_04pad.nc"
    if pad.exists():
        paths.append(pad)
    return paths

def build_season(season, in_dir=IN_DIR, out_dir=OUT_DIR):
    """Write one daily_sf_YYYYMMDD.nc per DJFM day of 'season', plus a data list"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.concat([open_sf(p) for p in season_files(season, in_dir)],
                   dim="valid_time").sortby("valid_time")
    sf = ds["sf"]

    if ACCUM_LABEL_IS_WINDOW_END:
        sf = sf.assign_coords(valid_time=sf.valid_time - pd.Timedelta(hours=1))

    # Guard: only keep days with a complete 24 hour window. A short day means a 
    # missing input month or a truncated pad, and would produce a low-biased
    # total that looks perfectly valid downstream.
    tidx = pd.DatetimeIndex(sf.valid_time.values)
    conv_day = (tidx - pd.Timedelta(hours=12)).normalize()
    per_day = tidx.to_series().groupby(conv_day).size()
    complete = pd.DatetimeIndex(per_day[per_day == 24].index) + pd.Timedelta(hours=12)

    # offset="12h" puts the bin edges on 12Z, so each bin is [12Z(D), 12Z(D+1))
    # -- the SPC convective day -- labelled 12Z(D). The month filter runs while
    # the label is still 12Z(D), so days are binned by their SPC day's month.
    daily = sf.resample(valid_time="1D", offset="12h").sum(skipna=False)
    daily = daily.sel(valid_time=daily.valid_time.isin(complete))
    daily = daily.sel(valid_time=daily.valid_time.dt.month.isin([12, 1, 2, 3]))

    # Re-stamp to the window MIDPOINT, 00Z(D+1). NodeFileFilter matches node time
    # to data time EXACTLY, so this is what makes Stage 2 use the 00Z node -- the
    # storm position half way through the window -- keeping the Hawcroft cap
    # centred on the storm. A cap anchored at 12Z(D) would trail a median storm
    # by ~9.7 deg by the window's end, about the 12 deg cap radius itself.
    daily = daily.assign_coords(valid_time=daily.valid_time + pd.Timedelta(hours=12))

    daily.attrs.update(
        units="m of water equivalent",
        long_name="Daily total snowfall, 12Z-12Z (SPC convective day)",
        temporal_method=("12Z-12Z SPC convective-day accumulation of ERA5 hourly sf; "
                         "timestamp is the 00Z window midpoint, filename is the SPC day"),
        source="ERA5 reanalysis-era5-single-levels, variable snowfall",
    )

    written = []
    for t in daily.valid_time.values:
        day = pd.Timestamp(t) - pd.Timedelta(hours=12)
        out = daily.sel(valid_time=[t]).to_dataset(name="sf")
        out = out.rename({"valid_time": "time"})
        out = out.drop_vars([c for c in DROP_COORDS if c in out.variables],
                            errors="ignore")
        # 0-360 to match the TE tracks and preprocess_era5_hourly.py
        out = out.assign_coords(longitude=(out["longitude"] % 360)).sortby("longitude")
        out.attrs["lon_convention"] = "0-360"

        # Self-describing for possible convective day compositing
        out.attrs["spc_convective_day"] = f"{day:%Y-%m-%d}"
        out.attrs["window_start"] = f"{day:%Y-%m-%d}T12:00Z"
        out.attrs["window_end"] = f"{day + pd.Timedelta(days=1):%Y-%m-%d}T12:00Z"

        path = out_dir / f"daily_sf_{day:%Y%m%d}.nc"
        tmp = path.with_suffix(".nc.tmp")
        out.to_netcdf(tmp)
        tmp.replace(path)
        written.append(path)

    lst = out_dir / f"in_data_list_dailysf_{season}.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in written) + "\n")
    print(f"season {season}: {len(written)} daily file(s) -> {lst}")
    return written 

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--season-min", type=int, default=1996)
    p.add_argument("--season-max", type=int, default=2025)
    p.add_argument("--in-dir", type=Path, default=IN_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()
    for s in range(a.season_min, a.season_max + 1):
        build_season(s, a.in_dir, a.out_dir)

