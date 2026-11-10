"""
download_era5_cores.py
======================
Downloads ERA5 data for each KDE outbreak core within every OC, over a box enclosing the 
KDE footprints of its kept outbreak cores, and saves those footprints 
as masks on the ERA5 0.25 grid so the environment can be subset to each
core afterwards.

Footprints
----------
Each kept core's footprint comes from the same sliding-window KDE that formed the cores 
(kde_cluster_track in ETCs_KDE.ipynb): 12h windows stepped by 1h, Gaussian bandwidth, 1 deg,
PDF threshold of 0.001), fitted to the core's OWN reports. The footprint is the union of 
every window's island holding a core report. For cores that were never gap-split this
reproduces the islands the engine itself saw; for gap-split cores, windows straddling the gap
no longer hand a sibling core's area to this one.

A single KDE over the core's whole lifetime (as previously drawn in maps) is NOT used:
the PDF is normalized by the number of reports, so for large cores the contour shrinks
and reports fall outside it. For small cores it is far from the cluster of reports. 

The KDE is evaluated directly on the ERA5 grid (grid origin snapped to multiples of DX), so mask
cells line up exactly with ERA5 points.

Domain download
---------------
One box per core: bounding box of the core's footprint plus a buffer BUFFER_DEG, snapped outward to the 0.25 grid. Hours run
from LEAD_HOURS before the core's first report to its last report. Masking 
to footprints happens after download, so the footprint definition can change without 
re-downloading.

Produces, in OUT_DIR:
    era5_sfc_core<core_id>_<YYYY-MM-DD>.nc
    era5_plev_core<core_id>_<YYYY-MM-DD>.nc
    ...
    manifest_cores.csv    (one row per (core, date): area, hours)

Usage:
    python download_era5_cores.py                     # dry run (footprints + manifest only)
    python download_era5_cores.py --download          # submits to CDS
    python download_era5_cores.py --download --workers 3 --only sfc
    python download_era5_cores.py --tracks 25 6117    # just a few OCs
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from sklearn.neighbors import KernelDensity

# reuse stuff from composite script
from download_era5_composite import DATASETS, _retrieve_one

# configuration
OUT_DIR = Path("/data1/lepique/era5_cores")
REPORTS = Path("/home1/lepique/circs_research/data/reports_tagged_final.csv")
CORES = Path("/home1/lepique/circs_research/data/kde_core_summary.csv")

# KDE setting
BANDWIDTH = 1.0
PDF_THRESH = 0.001
DX = 0.25
PAD = 2.0

WIN_HOURS = 12
STEP_HOURS = 1
MIN_REPORTS = 1       # window is analyzed only if it holds more than this # of reports

# footprint mask grid (inclusive bounds, degrees). Covers CONUS + KDE spread
GRID_LAT = (10.0, 65.0)
GRID_LON = (-135.0, -50.0)

# Download domain/time padding around each core's footprints
BUFFER_DEG = 3.0
LEAD_HOURS = 6

## ---------------------------------------------------------------------------
## footprints
## ---------------------------------------------------------------------------
def window_islands(g: pd.DataFrame,
                   bandwidth: float = BANDWIDTH,
                   pdf_thresh: float = PDF_THRESH,
                   dx: float = DX,
                   pad: float = PAD,
                   win_hours: int = WIN_HOURS,
                   step_hours: int = STEP_HOURS,
                   min_reports: int = MIN_REPORTS):
    """
    Sliding-window KDE islands for ONE core's reports, windowed exactly like kde_cluster_track
    
    Returns
    -------
    list of (win_start, iy, ix): window start time and the integer grid indices (lat / dx, lon / dx)
    of the island's cells, for every island that holds at least one report. Islands within o
    one window are disjoint.
    """

    g = g.sort_values('time_hour')
    lon = g['report_lon'].to_numpy()
    lat = g['report_lat'].to_numpy()
    times = g['time_hour']

    win = pd.Timedelta(hours=win_hours)
    step = pd.Timedelta(hours=step_hours)
    t_end = times.max()

    out = []
    start = times.min()
    while start <= t_end:
        in_win = ((times >= start) & (times < start + win)).to_numpy()
        win_start = start
        start = start + step
        if in_win.sum() <= min_reports:
            continue 

        xs, ys = lon[in_win], lat[in_win]

        ## padded bounding box, snapped outward to multiples of dx so every cell is an ERA grid point.
        ix0 = int(np.floor((xs.min() - pad) / dx))
        ix1 = int(np.ceil((xs.max() + pad) / dx))
        iy0 = int(np.floor((ys.min() - pad) / dx))
        iy1 = int(np.ceil((ys.max() + pad) / dx))
        GX, GY = np.meshgrid(np.arange(ix0, ix1 + 1) * dx,
                             np.arange(iy0, iy1 + 1) * dx)

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(np.column_stack([xs, ys]))
        dens = np.exp(kde.score_samples(np.column_stack([GX.ravel(), GY.ravel()])))
        dens = dens.reshape(GX.shape)

        labels, nlab = label(dens > pdf_thresh)
        if nlab == 0:
            continue 

        ## keep only islands that hold a report (nearest grid cell, as in the engine)
        rx = np.round(xs / dx).astype(int) - ix0
        ry = np.round(ys / dx).astype(int) - iy0
        for L in np.unique(labels[ry, rx]):
            if L == 0:
                continue 
            yy, xx = np.nonzero(labels == L)
            out.append((win_start, yy + iy0, xx + ix0))
    return out

def build_footprints(reports: pd.DataFrame, cores: pd.DataFrame):
    """
    Footprint of every core in ''cores'' on the fixed GRID_LAT x GRID_LON grid.
    
    Returns
    -------
    fp : xr.Dataset with n_windows (core_id, latitude, longitude) and 
    boxes : DataFrame, one row per core: footprint lat/lon bounds, n_cells,
        t_start, t_end, n_reports.
    """
    iy_lo, iy_hi = round(GRID_LAT[0] / DX), round(GRID_LAT[1] / DX)
    ix_lo, ix_hi = round(GRID_LON[0] / DX), round(GRID_LON[1] / DX)
    lat = np.arange(iy_lo, iy_hi + 1) * DX
    lon = np.arange(ix_lo, ix_hi + 1) * DX

    counts = np.zeros((len(cores), len(lat), len(lon)), dtype=np.int16)
    by_core = reports.groupby('core_id')
    rows = []
    for k, c in enumerate(cores.itertuples(index=False)):
        gc = by_core.get_group(c.core_id)
        n_clipped = 0
        for _, iy, ix in window_islands(gc):
            iy, ix = iy - iy_lo, ix - ix_lo
            ok = (iy >= 0) & (iy < len(lat)) & (ix >= 0) & (ix < len(lon))
            n_clipped += int((~ok).sum())
            counts[k, iy[ok], ix[ok]] += 1
        if n_clipped:
            print(f"  warning: {c.core_id} footprint clipped by mask grid "
                  f"({n_clipped} cell-windows)")

        yy, xx = np.nonzero(counts[k])
        rows.append({'core_id': c.core_id, 'track_id': c.track_id,
                     'n_reports': len(gc),
                     't_start': c.t_start, 't_end': c.t_end,
                     'lat_min': lat[yy.min()], 'lat_max': lat[yy.max()],
                     'lon_min': lon[xx.min()], 'lon_max': lon[xx.max()],
                     'n_cells': len(yy)})

    fp = xr.Dataset(
        {'n_windows': (('core_id', 'latitude', 'longitude'), counts,
                       {'long_name': 'number of sliding windows in which the cell '
                                     'lies inside one of the core KDE islands',
                        'note': 'n_windows > 0 is the core footprint'})},
        coords={'core_id': cores['core_id'].to_numpy().astype(str),
                'track_id': ('core_id', cores['track_id'].to_numpy()),
                'latitude': lat, 'longitude': lon},
        attrs={'bandwidth_deg': BANDWIDTH, 'pdf_thresh': PDF_THRESH,
               'dx_deg': DX, 'pad_deg': PAD, 'win_hours': WIN_HOURS,
               'step_hours': STEP_HOURS, 'min_reports': MIN_REPORTS},
    )
    return fp, pd.DataFrame(rows)

def footprint_mask(fp: xr.Dataset, core_id: str, like) -> xr.DataArray:
    """
    Boolean footprint of ``core_id`` aligned to an ERA5 field ``like``
    (any latitude order). Use as era5.where(footprint_mask(fp, cid, era5)).
    """
    m = fp['n_windows'].sel(core_id=core_id) > 0
    return m.reindex(latitude=like['latitude'], longitude=like['longitude'],
                     fill_value=False)

## ---------------------------------------------------------------------------
## download plan
## ---------------------------------------------------------------------------
def build_core_requests(boxes: pd.DataFrame, buffer_deg=BUFFER_DEG,
                        lead_hours=LEAD_HOURS) -> pd.DataFrame:
    """
    One row per (core, date): CDS area [N,W,S,E] enclosing the core's footprint plus 
    ''buffer_deg'', and the hours of theat date between ''lead_hours'' before the cores
    first report and its last report.
    """
    rows = []
    for b in boxes.itertuples(index=False):
        north = np.ceil((b.lat_max + buffer_deg) / DX) * DX
        south = np.floor((b.lat_min - buffer_deg) / DX) * DX
        west = np.floor((b.lon_min - buffer_deg) / DX) * DX
        east = np.ceil((b.lon_max + buffer_deg) / DX) * DX

        t0 = pd.Timestamp(b.t_start) - pd.Timedelta(hours=lead_hours)
        t1 = pd.Timestamp(b.t_end)
        hours = pd.date_range(t0, t1, freq='1h')

        for date, h in pd.Series(hours, index=hours).groupby(hours.strftime('%Y-%m-%d')):
            rows.append({'core_id': b.core_id, 'track_id': int(b.track_id), 'date': date,
                         'hours': h.dt.strftime('%H:00').tolist(),
                         'area': [float(north), float(west), float(south), float(east)]})

    return pd.DataFrame(rows)


def download(requests: pd.DataFrame, out_dir=OUT_DIR, dry_run=True, workers=1,
             only='both', overwrite=False):
    """
    Submit one CDS request per (core, date, dataset). Idempotent: existing
    outputs are skipped unless ``overwrite``. Atomic .part -> rename and zip
    unpacking come from download_era5_composite._retrieve_one.
    """
    out_dir = Path(out_dir)
    keys = ['sfc', 'plev'] if only == 'both' else [only]

    tasks = []
    for r in requests.itertuples(index=False):
        y, m, d = r.date.split('-')
        for key in keys:
            dataset, vars_, levels, prefix = DATASETS[key]
            target = out_dir / f"{prefix}_core{r.core_id}_{r.date}.nc"
            if target.exists() and not overwrite:
                print(f" [skip] {target.name}")
                continue
            request = {
                "product_type": "reanalysis",
                "variable": vars_,
                "year": y, "month": m, "day": d,
                "time": r.hours,
                "area": r.area,
                "data_format": "netcdf",
            }
            if levels:
                request["pressure_level"] = levels  # singular key (required)
            if dry_run:
                print(f" [dry] {target.name} hours={len(r.hours)} area={r.area}")
            else:
                tasks.append((dataset, request, target))

    if not dry_run and tasks and importlib.util.find_spec("xarray") is None:
        raise SystemExit("ERROR: xarray is needed to unpack CDS zip responses; "
                         "run this in the storm env.")

    if not dry_run and tasks:
        print(f" submitting {len(tasks)} request(s) with {workers} worker(s) ...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_retrieve_one, ds, rq, tg): tg for ds, rq, tg in tasks}
            for fut in as_completed(futs):
                tg = futs[fut]
                try:
                    print(f" -> saved {fut.result()}")
                except Exception as e:  # keep going; re-run retries this file
                    print(f" !! FAILED {tg.name}: {e}")


## ---------------------------------------------------------------------------
## main
## ---------------------------------------------------------------------------
def load_data(tracks=None):
    cores = pd.read_csv(CORES)
    cores = cores[cores['kept'] == True].copy()
    cores['track_id'] = cores['track_id'].astype(int)
    if tracks:
        cores = cores[cores['track_id'].isin(tracks)]
    cores = cores.sort_values(['track_id', 't_start']).reset_index(drop=True)

    reports = pd.read_csv(REPORTS, low_memory=False)
    reports = reports[reports['core_id'].isin(cores['core_id'])].copy()
    reports['time_hour'] = pd.to_datetime(reports['time_hour'], utc=True)
    return reports, cores


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=str(OUT_DIR))
    p.add_argument('--download', action='store_true',
                   help='actually submit to CDS (default: dry run)')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--only', default='both', choices=['both', 'sfc', 'plev'])
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--tracks', type=int, nargs='*', help='only these track_ids')
    p.add_argument('--buffer', type=float, default=BUFFER_DEG,
                   help=f'degrees added around the footprints (default {BUFFER_DEG})')
    p.add_argument('--lead-hours', type=int, default=LEAD_HOURS,
                   help=f'hours before first core onset (default {LEAD_HOURS})')

    args = p.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reports, cores = load_data(args.tracks)
    print(f"{len(cores)} kept cores across {cores['track_id'].nunique()} OCs")

    fp, boxes = build_footprints(reports, cores)
    # subset runs (--tracks) get their own file names so theye never overwrite
    # the full record footprints (don't plan to use that though, just in case)
    tag = '' if not args.tracks else '_tracks' + '-'.join(map(str, args.tracks))

    fp.to_netcdf(out_dir / f'core_footprints{tag}.nc',
                 encoding={'n_windows': {'zlib': True, 'complevel': 4}})
    
    boxes.to_csv(out_dir / f'core_boxes{tag}.csv', index=False)

    requests = build_core_requests(boxes, args.buffer, args.lead_hours)
    requests.to_csv(out_dir / f'manifest_cores{tag}.csv', index=False)

    n_keys = 1 if args.only != 'both' else 2

    span = requests.groupby('core_id')['area'].first().apply(
        lambda a: pd.Series({'lat_span': a[0] - a[2], 'lon_span': a[3] - a[1]}))
    print(f"{len(requests)} (core, date) pairs -> {n_keys * len(requests)} CDS requests; "
          f"{requests['hours'].str.len().sum()} core-hours")
    print(f"box size (deg), median: {span['lat_span'].median():.1f} lat x "
          f"{span['lon_span'].median():.1f} lon; max: {span['lat_span'].max():.1f} x "
          f"{span['lon_span'].max():.1f}")

    download(requests, out_dir, dry_run=not args.download, workers=args.workers,
             only=args.only, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
        
