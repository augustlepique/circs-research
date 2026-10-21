"""
download_era5_divergence.py
===========================
Download GLOBAL ERA5 upper-level divergence for the same timestamps as
``download_era5_composite.py``, as input for computing the irrotational
(divergent) wind with windspharm/pyspharm.

Why a separate, global download: the rotational/irrotational decomposition is
only unique on the full sphere, so the spherical-harmonic inversion
(chi = laplacian^-1(D), v_chi = grad(chi)) needs global fields -- the regional
``AREA`` used for the composite files will not do. Divergence alone suffices
(v_chi depends only on D), and since v_chi is a smooth, large-scale field a
0.5 deg grid loses nothing (~58 GB total vs ~230 GB at 0.25 deg). v_chi is later
interpolated onto the regional 0.25 deg grid of the era5_pv_ files.

Timestamps (cyclone min-MSL time +/- offsets, plus outbreak max-report hours)
are built with the SAME functions as download_era5_composite.py, so every
era5_divg_ file matches an era5_plev_ / era5_pv_ file date-for-date.

Produces, in ``OUT_DIR``:
  era5_divg_<YYYY-MM-DD>.nc   divergence d (s-1) at 300/250/200 hPa, global
                              0.5 deg, all needed hours of a date
  in_data_list_divg.txt       absolute paths, one per line, chronological

Usage
-----
    python download_era5_divergence.py                         # dry run
    python download_era5_divergence.py --download --workers 3  # submit to CDS
    python download_era5_divergence.py --download --season-min 2020 --season-max 2020

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
    timestamps_from_etc_summary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "reanalysis-era5-pressure-levels"
DIV_VARS = ["divergence"]                    # -> d  (s-1)
DIV_LEVELS = ["300", "250", "200"]           # 300-200 hPa outflow layer
GRID = [0.5, 0.5]                            # global, regridded by CDS
PREFIX = "era5_divg"
DATA_LIST = "in_data_list_divg.txt"


def build_request(date, hours):
    """Global divergence request for one date. No 'area' key -> full globe."""
    y, m, d = date.split("-")
    return {
        "product_type": "reanalysis",
        "variable": DIV_VARS,
        "pressure_level": DIV_LEVELS,  # singular key (required)
        "year": y,
        "month": m,
        "day": d,
        "time": hours,
        "grid": GRID,
        "data_format": "netcdf",
    }


def write_data_list(date_groups, out_dir):
    """Write in_data_list_divg.txt: absolute paths, chronological."""
    out_dir = Path(out_dir)
    lst = out_dir / DATA_LIST
    lst.write_text(
        "\n".join(str((out_dir / f"{PREFIX}_{d}.nc").resolve())
                  for d in sorted(date_groups))
        + "\n"
    )
    print(f"  data list (divg) -> {lst}  ({len(date_groups)} dates)")
    return lst


def download(date_groups, out_dir=OUT_DIR, dry_run=True, workers=1,
             overwrite=False, write_bookkeeping=True):
    """
    Submit one CDS request per date. Idempotent (existing files are skipped
    unless ``overwrite``) and atomic via ``_retrieve_one``'s .part->rename, so
    re-running after an interruption retries only what is missing or failed.
    ``write_bookkeeping`` must be False for season-filtered runs, for the same
    reason as in download_era5_composite.download().
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
            print(f" [dry] {target.name} hours={hours} levels={DIV_LEVELS} grid={GRID}")
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
        description="Download global ERA5 upper-level divergence for ETC compositing."
    )
    parser.add_argument("--etc-summary", default="data/etc_summary_final.csv",
                        help="CSV with 'track_id' and 'time_min_msl' columns.")
    parser.add_argument("--node-reports", default="data/nodes_with_reports_final.csv",
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
    download(build_date_groups(manifest), out_dir=Path(args.out_dir),
             dry_run=not args.download, workers=args.workers,
             overwrite=args.overwrite, write_bookkeeping=not season_filtered)
