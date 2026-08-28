"""
download_nohrsc.py
==================
Disclaimer: Made this script with help from AI

Stage 1b of the ETC snowfall pipeline: acquire the NOHRSC National Gridded
Snowfall Analysis (v2) as a parallel overlay to the ERA5 'sf'.

Two phases, run independently:
 --download     # fetch raw 24h NOHRSC netCDFs into RAW_DIR (cached, resumable)
 --regrid       # conservatively regrid them onto the ERA5 daily grid (for TE use)

 
Window Alignment
----------------
NOHRSC 24-h products are stamped at 12Z and cover the 24 hours ENDING at that stamp.
The SPC convective day runs 12Z(D) -> 12Z(D+1), so the file for day D is the one
stamped 12Z on D+1 -- and lives under that date's YYYYMM directory. 

The regridded output is stamped 00Z(D+1), the window MIDPOINT, matching build_daily_sf.py
so the same 00Z stage 2 nodefile drives both products.

UNITS (not interchangeable with ERA5)
NOHRSC data is snowfall accumulation depth in meters. ERA5 'sf' is water equivalent
in meters. Output variable is named 'snowfall_depth' rather than sf to avoid confusion.

Source grid: 0.04 deg regular lat/lon, 850 x 1500, lat 55->21, lon -126->-66,
with lat_bounds/lon_bounds supplied (used for regridding).

Record begins in winter season 2009. 

Usage
-----
    python download_nohrsc.py --download        # all seasons
    python download_nohrsc.py --download --season-min 2020 --season-max 2020   # just one season
    python download_nohrsc.py --regrid         # after downloading
    python download_nohrsc.py --download --regrid --workers 3
"""

import argparse
import csv 
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path 

import numpy as np
import pandas as pd
import requests   # for http library
import xarray as xr
import xesmf as xe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path("/data1/lepique/nohrsc_cache/")        # reuses existing cache
OUT_DIR = Path("/data1/lepique/nohrsc_daily/")
WEIGHTS = Path("/data1/lepique/nohrsc_daily/nohrsc_to_era5_conservative.nc")
MISS_LOG = Path("/data1/lepique/nohrsc_daily/nohrsc_misses.csv")

# an existing ERA5 daily file defines the target grid, so the two products are guaranteed
# to share it rather than being independently reconstructed.
ERA5_GRID_REF = Path("/data1/lepique/era5_sf_daily/daily_sf_20200118.nc")

URL_TMPL = ("https://www.nohrsc.noaa.gov/snowfall_v2/data/"
            "{ym}/sfav2_CONUS_24h_{stamp}.nc")
RAW_TMPL = "nohrsc_{stamp}.nc"  # match existing cache

SEASON_MIN, SEASON_MAX = 2009, 2025

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0
TIMEOUT = 120
USER_AGENT = "CIRCS_research ETC-snowfall pipeline (contact: lepique@wisc.edu)"

# Fraction of a target cell allowed to be NaN before it is masked out entirely,
# 0.5 => keep a cell only if at least half of it is covered by analyzed source data.
NA_THRES = 0.5

# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def spc_days(season):
    """
    SPC convective days D making up DJFM 'season' (labeled by its JFM year:
    season 2020 = Dec 2019 + JFM 2020). Matches build_daily_sf.py's day set.
    """
    return pd.date_range(f"{season - 1}-12-01", f"{season}-03-31", freq="D")

def stamp_for(day):
    """
    NOHRSC valid stamp (YYYYMMDDHH) for SPC convective day 'day'
    
    The 24h analysis covers the period ending at its stamp, so day D's window 
    12Z(D) -> 12Z(D+1) is the file stamped 12Z on D+1
    """
    return f"{pd.Timestamp(day) + pd.Timedelta(days=1):%Y%m%d}12"

def url_for(stamp):
    return URL_TMPL.format(ym=stamp[:6], stamp=stamp)

def raw_path(stamp, raw_dir=RAW_DIR):
    return Path(raw_dir) / RAW_TMPL.format(stamp=stamp)

def enumerate_needed(season_min=SEASON_MIN, season_max=SEASON_MAX):
    """[(spc_day, stamp), ...] in chronological order, deduplicated."""
    out=[]
    seen = set()
    for s in range(season_min, season_max + 1):
        for d in spc_days(s):
            st = stamp_for(d)
            if st not in seen:
                seen.add(st)
                out.append((d, st))
    return out 

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def validate(path):
    """
    True if 'path' is a readable NOHRSC netCDF with the expected payload.
    
    checking the bytes open as netCDF is not enough: a truncated download can
    still parse a header. Opening the variable and checking shape verifies the file
    is complete
    """
    try:
        with xr.open_dataset(path) as ds:
            return "Data" in ds and ds["Data"].shape == (850, 1500)
    except Exception:
        return False 

def fetch_one(stamp, raw_dir=RAW_DIR, session=None):
    """
    Download one NOHRSC file with retry/backoff
    
    Returns (stamp, status) where status is "ok" | "cached" | "missing" | "failed":<reason>.
    A 404 is a permanent miss (does not exist) and is not retried. Others are.
    
    Writes to a .part sidecar and renames after validation.
    """
    target = raw_path(stamp, raw_dir)
    if target.exists():
        if validate(target):
            return stamp, "cached"
        target.unlink()               ## corrupted/truncated - refetch

    url = url_for(stamp)
    sess = session or requests 
    tmp = target.with_suffix(".nc.part")
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = sess.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
            if r.status_code == 404:
                return stamp, "missing"
            r.raise_for_status()
            tmp.write_bytes(r.content)
            if not validate(tmp):
                tmp.unlink(missing_ok=True)
                raise ValueError("downloaded file failed validation")
            tmp.replace(target)
            return stamp, "ok"
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS:
                return stamp, f"failed:{type(exc).__name__}"
            # exponential backoff with jitter, so parallel workers do not
            # retry in lockstep against the same server
            time.sleep(BACKOFF_BASE * 2 ** (attempt - 1) * (1 + random.random()))


def download(season_min=SEASON_MIN, season_max=SEASON_MAX, raw_dir=RAW_DIR,
             workers=2, miss_log=MISS_LOG):
    """
    Fetch every file needed, Idempotent (skips already existing files).
    Workers is kept at 2, maybe 3 at most b/c this is a public server and archive is not rate-limit documented
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    Path(miss_log).parent.mkdir(parents=True, exist_ok=True)

    needed = enumerate_needed(season_min, season_max)
    print(f"seasons {season_min}-{season_max}: {len(needed)} NOHRSC files needed")

    results={}
    with requests.Session() as sess, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_one, st, raw_dir, sess): (d, st)
                for d, st in needed}
        for i, fut in enumerate(as_completed(futs), 1):
            day, stamp = futs[fut]
            _, status = fut.result()
            results[stamp] = (day, status)
            if i % 200 == 0:
                print(f" ...{i}/{len(needed)}")


    counts = {}
    for _, (_, s) in results.items():  # result.items() gives (stamp, (day, status))
        counts[s.split(":")[0]] = counts.get(s.split(":")[0], 0) + 1
    print("\n====== download summary ======")
    for k in sorted(counts):
        print(f"  {k:10s}: {counts[k]}")

    misses = [(d, st, s) for st, (d, s) in sorted(results.items())
              if s not in ("ok", "cached")]
    with open(miss_log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spc_convective_day", "nohrsc_stamp", "url", "status"])
        for d, st, s in misses:
            w.writerow([f"{d:%Y-%m-%d}", st, url_for(st), s])
    print(f" miss log -> {miss_log}. ({len(misses)} entries)")
    return results 

# ---------------------------------------------------------------------------
# Regrid
# ---------------------------------------------------------------------------
def _src_grid(ds):
    """Source grid with 1-D cell edges taken from NOHRSC's own bounds."""
    lat_b = np.append(ds.lat_bounds[:, 0].values, ds.lat_bounds[-1, 1].values)
    lon_b = np.append(ds.lon_bounds[:, 0].values, ds.lon_bounds[-1, 1].values) % 360
    lon = ds.lon.values % 360
    if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lon_b) > 0)):
        raise SystemExit("ERROR: NOHRSC longitudes are not monotonic after the "
                         "0-360 conversion (domain wraps?); regrid weights would "
                         "be silently wrong")
    return xr.Dataset({"lat": ds.lat, "lon": ("lon", lon),
                       "lat_b": ("lat_b", lat_b), "lon_b": ("lon_b", lon_b)})


def _tgt_grid(ref=ERA5_GRID_REF):
    """Target grid read from an ERA5 daily file so the two products match"""
    era = xr.open_dataset(ref)
    lat, lon = era.latitude.values, era.longitude.values

    def edges(c):   # ERA5 data doesn't have edges natively so we need to infer them
        d = np.diff(c).mean()
        return np.append(c - d / 2, c[-1] + d / 2)
    era.close()

    return xr.Dataset({"lat": ("lat", lat), "lon": ("lon", lon),
                       "lat_b": ("lat_b", edges(lat)),
                       "lon_b": ("lon_b", edges(lon))})

def build_regridder(sample, ref=ERA5_GRID_REF, weights=None):
    """
    Conservative NOHRSC -> ERA5 regridder, with weights cached to disk."""
    tgt = _tgt_grid(ref)
    src = _src_grid(sample)
    if weights is None:
        # Signature the cache on both grids, so a change to either produces a new
        # filename instead of silently reusing weights that no longer apply.
        sig = (f"{src.sizes['lon']}x{src.sizes['lat']}"
                f"_{float(src.lon[0]):.2f}-{float(src.lon[-1]):.2f}"
                f"__{tgt.sizes['lon']}x{tgt.sizes['lat']}"
                f"_{float(tgt.lon[0]):.2f}-{float(tgt.lon[-1]):.2f}")
        weights = WEIGHTS.with_name(f"nohrsc_to_era5_conservative_{sig}.nc")
    Path(weights).parent.mkdir(parents=True, exist_ok=True)
    return xe.Regridder(src, tgt, "conservative",
                        ignore_degenerate=True,
                        filename=str(weights),
                        reuse_weights=Path(weights).exists())

def regrid_season(season, raw_dir=RAW_DIR, out_dir=OUT_DIR, ref=ERA5_GRID_REF,
                  regridder=None, na_thres=NA_THRES):
    """Write one daily_snow_nohrsc_YYYYMMDD.nc per available SPC day, plus a data list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    #_, era = _tgt_grid(ref)

    days = [(d, stamp_for(d)) for d in spc_days(season)]
    days = [(d, st) for d, st in days if raw_path(st, raw_dir).exists()]
    if not days:
        print(f"season {season}: no raw files present, skipping")
        return []
    if regridder is None:
        with xr.open_dataset(raw_path(days[0][1], raw_dir)) as s0:
            regridder = build_regridder(s0, ref)

    written = []
    for day, stamp in days:
        src = xr.open_dataset(raw_path(stamp, raw_dir))
        # Guard: analysis grid must not drift between eras. 
        if src["Data"].shape !=(850, 1500):
            raise SystemExit(f"ERROR: {stamp} has grid {src['Data'].shape}, "
                             f"expected (850, 1500); cached weights are invalid")

        da = src["Data"]
        da = da.assign_coords(lon=da.lon % 360)   # same 0-360 convention as _src_grid
        out = regridder(da.where(np.isfinite(da)), skipna=True, na_thres=na_thres)

        # Stamp at the window midpoint, 00Z(D+1)
        t = pd.Timestamp(day) + pd.Timedelta(hours=24)
        ds = out.rename("snowfall_depth").to_dataset()
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        ds = ds.expand_dims(time=[t])

        ds["snowfall_depth"].attrs.update(
            units="m",
            long_name="24-h snowfall accumulation depth (NOT water equivalent)",
            temporal_method=("12Z-12Z SPC convective-day accumulation; timestamp is "
                             "the 00Z window midpoint, filename is the SPC day"),
            source="NOHRSC National Gridded Snowfall Analysis v2, 24-h product",
            regrid_method="conservative (xesmf), 0.04 deg -> ERA5 0.25 deg",
            note=("snowfall DEPTH, not water equivalent; not directly comparable "
                  "to ERA5 sf without a snow-to-liquid-ratio conversion"),
        )
        ds.attrs.update(
            lon_convention="0-360",
            spc_convective_day=f"{day:%Y-%m-%d}",
            window_start=f"{day:%Y-%m-%d}T12:00Z",
            window_end=f"{day + pd.Timedelta(days=1):%Y-%m-%d}T12:00Z",
            nohrsc_source_file=RAW_TMPL.format(stamp=stamp),
        )

        path = out_dir / f"daily_snow_nohrsc_{day:%Y%m%d}.nc"
        tmp = path.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp)
        tmp.replace(path)
        written.append(path)
        src.close()

    lst = out_dir / f"in_data_list_nohrsc_{season}.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in written) + "\n")
    print(f"season {season}: {len(written)} daily file(s) -> {lst}")
    return written

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true")
    p.add_argument("--regrid", action="store_true")
    p.add_argument("--season-min", type=int, default=SEASON_MIN)
    p.add_argument("--season-max", type=int, default=SEASON_MAX)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    if not (a.download or a.regrid):
        p.error("pass --download and/or --regrid")

    if a.download:
        download(a.season_min, a.season_max, a.raw_dir, a.workers)

    if a.regrid:
        rg = None
        for s in range(a.season_min, a.season_max + 1):
            first = next((st for _, st in
                          [(d, stamp_for(d)) for d in spc_days(s)]
                          if raw_path(st, a.raw_dir).exists()), None)
            if rg is None and first:
                with xr.open_dataset(raw_path(first, a.raw_dir)) as s0:
                    rg = build_regridder(s0)
            regrid_season(s, a.raw_dir, a.out_dir, regridder=rg)


    






