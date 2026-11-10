"""
download_era5_uv_global.py
==========================
Download GLOBAL ERA5 upper level u and v winds for the same timestamps as
the composite downloads, as input for computing the irrotational (divergent) wind
with windspharm.

Why global: the split of the wind into rotational/irrotational parts is only unique
on the full sphere nd windspharm works with spherical harmonics, which needs a
global grid. A 0.5 degree grid is sufficient since irrotational wind field is 
smooth (and saves lots of space). It is later interpolated onto the regional 
0.25 degree grid for compositing.

Timestamps are the union of both composite sets (min msl reference and core onset)
Existing fiiles are skipped, so if needed_times.csv later gains hours on a date
that already has a file, re=download those dates with --overwrite -- otherwise
they won't download.

Produces, in ''OUT_DIR'':
    era5_uv_global_<YYYY-MM-DD>.nc  u, v (m/s) at 300/250/200 hPa, global 0.5 deg
                                    all needed hours of a date
    in_data_list_uv_global.txt      absolute paths, one per line, chronological 
                                    (for TE NodeFileCompose)

                                    
Usage
-----
    python dowwnload_era5_uv_global.py        # dry run
    python download_era5_uv_global.py --download --workers 3  # submit to CDS
    python download_era5_uv_global.py --download --season-min 2020 --season-max 2020
    
    DISCLAIMER: Written with help from claude. 
"""

import argparse
from pathlib import Path

import pandas as pd

from download_era5_composite import (
    COL_TIMEMAXREP,
    OUT_DIR,
    _retrieve_one,
    add_max_report_time,
    build_date_groups,
    filter_by_season,
    season_year,
    timestamps_from_etc_summary,
)
from download_era5_onset import NEEDED, needed_hours

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "reanalysis-era5-pressure-levels"
UV_VARS = ["u_component_of_wind",            # -> u  (m s-1)
           "v_component_of_wind"]            # -> v  (m s-1)
UV_LEVELS = ["300", "250", "200"]           # 300-200 hPa outflow layer
GRID = [0.5, 0.5]                            # global, lower resolution than 0.25, 0.25 (for storage)
PREFIX = "era5_uv_global"
DATA_LIST = "in_data_list_uv_global.txt"

def build_request(date, hours):
    """Global u, v request for one date. No 'area' key since its for the full globe"""
    y, m, d = date.split("-")
    return {
        "product_type": "reanalysis",
        "variable": UV_VARS,
        "pressure_level": UV_LEVELS,  # singular key (required)
        "year": y,
        "month": m,
        "day": d,
        "time": hours,
        "grid": GRID,
        "data_format": "netcdf",
    }

def onset_manifest(needed_csv, season_min=None, season_max=None):
    """(date, hour) rows for the onset/peak lag times, season-filtered like 
    filter_by_season."""
    need = needed_hours(needed_csv)
    sy = need.apply(season_year)
    if season_min is not None:
        need = need[sy >= season_min]
    if season_max is not None:
        need = need[sy <= season_max]
    return pd.DataFrame({"date": need.dt.strftime("%Y-%m-%d"),
                         "hour": need.dt.strftime("%H:00")})


def write_data_list(date_groups, out_dir):
    """Write in_data_list_uv_global.txt: absolute paths, chronological."""
    out_dir = Path(out_dir)
    lst = out_dir / DATA_LIST
    lst.write_text(
        "\n".join(str((out_dir / f"{PREFIX}_{d}.nc").resolve())
                  for d in sorted(date_groups))
        + "\n"
    )
    print(f"  data list (uv_global) -> {lst}  ({len(date_groups)} dates)")
    return lst

def download(date_groups, out_dir = OUT_DIR, dry_run=True, workers=1, 
             overwrite=False, write_bookkeeping=True):
    """
    Submit one CDS request per date. existing files are skipped unless ''overwrite''
    and atomic via ''_retrieve_one'''s .part->rename, so can re-run after an interrupted
    downloa. ''write_bookkeeping'' must be False for season-filtered downloads,
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for date, hours in sorted(date_groups.items()):
        target = out_dir / f"{PREFIX}_{date}.nc"
        if target.exists() and not overwrite:
            print(f" [skip] {target.name}")
            continue 
        if dry_run:
            print(f" [dry] {target.name} hours={hours} levels={UV_LEVELS} grid={GRID}")
        else:
            tasks.append((build_request(date, hours), target))

    if not dry_run and tasks:
        print(f" submitting {len(tasks)} request(s) with {workers} worker(s) ...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_retrieve_one, DATASET, rq, tg): tg for rq, tg in tasks}
            for fut in as_completed(futs):
                tg = futs[fut]
                try:
                    print(f" -> saved {fut.result()}")
                except Exception as e:  # keep going; re-run retries this date
                    print(f" !! FAILED {tg.name}: {e}")

    if write_bookkeeping:
        write_data_list(date_groups, out_dir)
    else:
        print(f"  [bookkeeping skipped: partial run] {DATA_LIST} left untouched.")
    print(f"\n  unique dates: {len(date_groups)}  "
          f"(one global request per date, {len(tasks)} submitted)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download global ERA5 upper level u, v winds for Helmholtz decomposition " \
        "and ETC compositing."
    )
    parser.add_argument("--etc-summary", default="/home1/lepique/circs_research/data/etc_summary_final.csv",
                        help="CSV with 'track_id' and 'time_min_msl' columns.")
    parser.add_argument("--node-reports", default="/home1/lepique/circs_research/data/nodes_with_reports_final.csv",
                        help="Per-node report counts, used for the max-report hour "
                             "when 'time_max_reports' is absent from --etc-summary.")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR}).")
    parser.add_argument("--download", action="store_true",
                        help="Actually submit to CDS. Default is a dry run.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent CDS requests (keep small, ~3-4).")
    parser.add_argument("--season-min", type=int, default=None,
                        help="Only cyclones whose DJFM season year is >= this.")
    parser.add_argument("--season-max", type=int, default=None,
                        help="Only cyclones whose DJFM season year is <= this.")
    parser.add_argument("--needed", default=str(NEEDED),
                        help=f"Onset/peak needed_times.csv (default: {NEEDED}).")
    parser.add_argument("--no-onset", action="store_true",
                        help="Skip the onset/peak hours (min-MSL times only).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download targets that already exist.")
    args = parser.parse_args()

    etc_summary = pd.read_csv(args.etc_summary)
    if COL_TIMEMAXREP not in etc_summary.columns and args.node_reports:
        etc_summary = add_max_report_time(etc_summary, pd.read_csv(args.node_reports))
    etc_summary = filter_by_season(etc_summary, args.season_min, args.season_max)
    season_filtered = args.season_min is not None or args.season_max is not None
    if season_filtered:
        print(f"Season filter {args.season_min}-{args.season_max}: "
              f"{len(etc_summary)} cyclones selected.")

    manifest, _ = timestamps_from_etc_summary(etc_summary)
    manifest = manifest[["date", "hour"]]
    n_msl = build_date_groups(manifest)
    if not args.no_onset:
        onset = onset_manifest(args.needed, args.season_min, args.season_max)
        manifest = pd.concat([manifest, onset]).drop_duplicates()
        n_onset = build_date_groups(onset)
        print(f"min-MSL: {len(n_msl)} dates | onset: {len(n_onset)} dates "
              f"({len(set(n_onset) - set(n_msl))} not in min-MSL) | "
              f"union: {len(manifest)} hours")
    download(build_date_groups(manifest), out_dir=Path(args.out_dir),
             dry_run=not args.download, workers=args.workers,
             overwrite=args.overwrite, write_bookkeeping=not season_filtered)
