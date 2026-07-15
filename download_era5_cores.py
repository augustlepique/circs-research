"""
download_era5_cores.py
======================
downloads ERA5 data for each kept outbreak core, spatially subset to the core's
own KDE footprint rather than a fixed domain.

Each core gets its own bounding box, computed from a 2D Gaussian KDE over that core's reports, thresholded at 
either 0.001 or 0.005 (TBD), then snapped to the ERA5 0.25 grid.

Produces, in OUT_DIR:
    era5_sfc_core<core_id>_<YYYY-MM-DD>.nc
    era5_plev_core<core_id>_<YYYY-MM-DD>.nc
    manifest_cores.csv.  one row per (core, date): area, hours, n_reports

Usage:
    python download_era5_cores.py. # dry run
    python download_era5_cores.py --download  # submits to CDS
    python download_era5_cores.py --download --workers 3 --only sfc # choose workers and surface/pressure level data
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

# reuse what we can from composite script 
from download_era5_composite import (
    SFC_VARS, PLEV_VARS, PRESSURE_LEVELS, _retrieve_one
)

# configuration
OUT_DIR = Path("/data1/lepique/era5_cores")

# KDE footprint (per core)


