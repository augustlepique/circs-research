"""
download_nohrsc.py
==================
Stage 1b of the ETC snowfall pipeline: acqurie the NOHRSC National Gridded
Snowfall Analysis (v2) as a prallel overlay to the ERA5 'sf'.

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path("/data1/lepique/nohrsc_cache/")        # reuses existing cache
OUT_DIR = Path("/data1/lepique/nohrsc_daily/")
WEIGHTS = Path("/data1/lepique/nohrsc_daily/nohrsc_to_era5_conservative.nc")
MISS_LOG = Path("/data1/lepique/nohrsc_daily/nohrsc_misses.csv")

# an existing ERA5 daily file defines the target frid, so the two products are guaranteed
# to share it rather than being independently reconstructed.
ERA5_GRID_REF = Path("/data1/lepique/era5_sf_daily/daily_sf_20200118.nc")

URL_TMPL = ("https://www.nohrsc.noaa.gov/snowfall_v2/data/"
            "{ym}/sfav2_CONUS_24h_{stamp}.nc")
RAW_TMPL = "nohrsc_{stamp}.nc"  # match existing cache

SEASON_MIN, SEASON_MAX = 2009, 2025
SEASON_MONTHS = (12, 1, 2, 3)

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0
TIMEOUT = 120
USER_AGENT = "CIRCS_research ETC-snowfall pipeline (contact: lepique@wisc.edu)"

# Fraction ofa target cell allowed to be NaN before it is masked out entirely,
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
    
    The 24h analysis covers the period ending at its stamp, so dau D's window 
    12Z(D) -> 12Z(D+1) is the file stamped 12Z on D+1
    """
    return f"{pd.Timestamp(day) + pd.Timedelta(days=1):%Y%m%d}12"

def url_for(stamp):
    return URL_TMPL.format(ym=stamp[:6], stamp=stamp)

def raw_path(stamp, raw_dir=RAW_DIR):
    return Path(raw_dir) / RAW_TMPL.format(stamp=stamp)

def enumerate_needed(season_min=SEASON_MIN, season_max=SEASON_MAX):
    """"""
