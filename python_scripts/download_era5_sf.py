"""
download_era5_sf.py
Stage 1 of te ETC snowfall pipeline: acquire contiguous hourly ERA5 
snowfall for DJFM 1995/96-2024/25.

This .py file was made (rather than reusing era5_sfcx_*.nc) because those files
were downloaded for NodeFileCompose and contain only each date's composite 
reference hours.  a 12UTC-centered daily accumulation needs all 24 hourly steps.
This script downloads the full hourly record instead.

Batching: one CDS request per (year,month) over DJFM. 

Produces, in OUT_DIR:
    era5_sf_<YYYY>_<MM>.nc  hourly sf for that month, whole domain
    era5_sf_<YYYY>_04pad.nc. Apr 1 hourly sf 

Usage
-----
    python download_era5_sf.py     # dry run
    python download_era5_sf.py --download --workers 3   # submit to CDS
    python download_era5_sf.py --download --season-min 2020 --season-max-2020
    python download_era5_sf.py --download --no-pad   # skips Apr 1 pads
"""

import argparse
import calendar
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the thread-local client + atomic .part->rename from the composite
# downloader.
from download_era5_composite import _get_client, _retrieve_one 

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path("/data1/lepique/era5_sf_hourly/")


# ERA5 domain [N, W, S, E]. CONUS plus margin for cyclones whose 12 deg cap
# reaches in from offshore. NOTE: intentionally tighter than the composite
# AREA ([70,-140,-10,-50]) -- these grids do NOT match, so era5_sf_* and
# era5_sfc_* cannot be stacked without a regrid.
AREA = [61, -144, 12, -47]

# Seasons to fetch, labelled by JFM year (season 2020 = Dec 2019 + JFM 2020),
# matching download_era5_composite.season_year().
SEASON_MIN = 1996
SEASON_MAX = 2025

SF_VAR = "snowfall"          # -> sf (m water equivalent), accumulated
DATASET = "reanalysis-era5-single-levels"
FILE_PREFIX = "era5_sf"

# ERA5 hourly accumulations are stamped at the END of their window: the field
# at valid_time t covers (t-1h, t]. Under that convention the daily total for
# 31 March needs 00Z 1 April, so we pad one day past each season.
# VERIFY THIS ON THE PILOT -- if the convention turns out to be [t, t+1), set
# ACCUM_LABEL_IS_WINDOW_END = False in build_daily_sf.py and the pad is unused
# (harmless, already downloaded).
PAD_APRIL1 = True

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# ---------------------------------------------------------------------------
# Request planning
# ---------------------------------------------------------------------------
def season_months(season):
    """
    Yield the (year, month) pairs making up DJFM season, where season is
    labelled by its JFM year
    """
    yield (season - 1, 12)
    for m in (1,2,3):
        yield (season, m)

def build_tasks(season_min=SEASON_MIN, season_max=SEASON_MAX,
                out_dir=OUT_DIR, area=AREA, pad=PAD_APRIL1):
    """
    Build the full list of (target_path, request_dict) pairs.
    
    One request per (year, month) covering every day and all 24 hours, plus one
    Apr-1 request per season when 'pad' is set. Returned in chronological order
    so a partial run leaves a contiguous prefix on disk.
    """
    out_dir = Path(out_dir)
    tasks=[]
    for season in range(season_min, season_max + 1):
        for year, month in season_months(season):
            ndays = calendar.monthrange(year, month)[1]
            target = out_dir / f"{FILE_PREFIX}_{year}_{month:02d}.nc"
            tasks.append((target, {
                "product_type": "reanalysis",
                "variable": [SF_VAR],
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, ndays + 1)],
                "time": ALL_HOURS,
                "area": area,
                "data_format": "netcdf",
            }))
        if pad:
            ## Oly 00Z is strictly needed, but a full day is ~1/30th of a month
            ## request and covers either accumulation convention.
            target = out_dir / f"{FILE_PREFIX}_{season}_04pad.nc"
            tasks.append((target, {
                "product_type": "reanalysis",
                "variable": [SF_VAR],
                "year": str(season),
                "month": "04",
                "day": ["01"],
                "time": ALL_HOURS,
                "area": area,
                "data_format": "netcdf",
            }))
    return tasks

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(tasks, dry_run=True, workers=1, overwrite=False):
    """
    Submit tasks to CDS.
    Existing target is skipped unless 'overwrite'.
    _retrieve_one writes to a .part sidecar and renames on success,
    so an interrupted request can never leave a truncated file.

    workers: CDS-Beta throttles concurrent requests, so this helps avoid wait time
    """
    pending=[]
    for target, request in tasks:
        if target.exists() and not overwrite:
            print(f" [skip] {target.name}")
            continue
        pending.append((target,request))
    
    print(f"\n{len(pending)} request(s) to submit "
          f"({len(tasks) - len(pending)} already on disk)")

    if dry_run:
        for target, request in pending:
            print(f" [dry] {target.name}  days={len(request['day'])}"
                  f"hours={len(request['time'])} area={request['area']}")
            print("\nDry run -- nothing submitted.")
            return []

    tasks[0][0].parent.mkdir(parents=True, exist_ok=True)
    done=[]
    if workers == 1:
        for target, request in pending:
            print(f" requesting {target.name} ...")
            _retrieve_one(DATASET, request, target)
            print(f" -> saved {target.name}")
            done.append(target)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_retrieve_one, DATASET, req, tgt): tgt
                    for tgt, req in pending}
            for fut in as_completed(futs):
                tgt = futs[fut]
                try:
                    fut.result()
                    print(f" -> saved {tgt.name}")
                    done.append(tgt)
                except Exception as exc:    # keep run alive
                    print(f" !! FAILED {tgt.name}: {exc}")
    return done

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true",
                   help="actually submit to CDS (default is a dry run)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--season-min", type=int, default=SEASON_MIN)
    p.add_argument("--season-max", type=int, default=SEASON_MAX)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-pad", action="store_true",
                   help="skip the Apr-1 pad requests")
    args = p.parse_args()

    tasks = build_tasks(args.season_min, args.season_max,
                        out_dir=args.out_dir, pad=not args.no_pad)
    print(f"seasons {args.season_min}-{args.season_max}: {len(tasks)} request(s)")
    download(tasks, dry_run=not args.download,
             workers=args.workers, overwrite=args.overwrite)







