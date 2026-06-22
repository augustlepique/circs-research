"""
download_era5_composite.py
==========================
Downloads ERA5 data for single-snapshot ETC composites

Produces:
  era5_composite_2020/
    era5_sfc_YYYY-MM-DD.nc       (MSLP, TCWV, T2m — single-level)
    era5_plev_YYYY-MM-DD.nc      (T, u, v, q @ 850 hPa — pressure-level)
    in_data_list_sfc.txt         (one path per line — feed to NodeFileCompose)
    in_data_list_plev.txt

Usage
-----
From the command line:
    python download_era5_composite.py            # dry run (default)
    python download_era5_composite.py --download # actually submit to CDS

From a notebook:
    from download_era5_composite import timestamps_from_picks, build_manifest, download, OUT_DIR
    timestamps  = timestamps_from_picks(picks_df)
    date_groups = build_manifest(timestamps)
    download(date_groups, OUT_DIR, dry_run=False)
"""

import argparse
from pathlib import Path

import cdsapi
import pandas as pd

OUT_DIR = Path("era5_composite_2020")

## ERA5 domain [N, W, S, E]
## extended south to cover nodes near the southern edge of CONUS
AREA = [70, -140, -10, -50]

SFC_VARS = [
    "mean_sea_level_pressure",    # -> msl   (Pa)
    "total_column_water_vapour",  # -> tcwv  (kg m-2)
    "2m_temperature",             # -> t2m   (K)
]

PLEV_VARS = [
    "temperature",                # -> t     (K)
    "u_component_of_wind",        # -> u     (m s-1)
    "v_component_of_wind",        # -> v     (m s-1)
    "specific_humidity",          # -> q     (kg kg-1)
]

PRESSURE_LEVELS = ["850"]         # add 250 later for upper-level jet
# Hardcoded fallback — only used when running as __main__ without picks_df.
# Leave empty if you always call timestamps_from_picks() from a notebook.
TIMESTAMPS = []

## build per date groups (batch same date hours into one CDS request)
def build_manifest(timestamps):
    """
    timestamps: list of ("YYYY-MM-DD", "HH:00") tuples
    Returns: dict date_str -> sorted list of hour strings
    """
    df = pd.DataFrame(timestamps, columns=["date", "hour"])
    return (
        df.groupby("date")["hour"]
        .apply(lambda h: sorted(h.unique().tolist()))
        .to_dict()
    )

    ## write in_data_list files (chronologically sorted absolute paths)
def write_data_lists(date_groups, out_dir):
    """
    Writes two text files listing the absolute paths to the downloaded NetCDF
    files, one per line, sorted chronologically.  These are passed directly to
    NodeFileCompose via --in_data_list.
    """
    sfc_paths  = sorted(out_dir / f"era5_sfc_{d}.nc"  for d in date_groups)
    plev_paths = sorted(out_dir / f"era5_plev_{d}.nc" for d in date_groups)

    sfc_list  = out_dir / "in_data_list_sfc.txt"
    plev_list = out_dir / "in_data_list_plev.txt"

    sfc_list.write_text(
        "\n".join(str(p.resolve()) for p in sfc_paths) + "\n"
    )
    plev_list.write_text(
        "\n".join(str(p.resolve()) for p in plev_paths) + "\n"
    )

    print(f"  data list (sfc)  -> {sfc_list}")
    print(f"  data list (plev) -> {plev_list}")
    return sfc_list, plev_list

## Download
def download(date_groups, out_dir=OUT_DIR, dry_run=True):
    """
    Submit CDS requests for all dates in date_groups.
    Skips files that already exist (safe to re-run after interruption).
    Always writes/refreshes the in_data_list files, even in dry-run mode.

    Parameters
    ----------
    date_groups : dict  date_str -> [hour_str, ...]
    out_dir     : Path
    dry_run     : bool  — if True, prints what would be requested but does
                          not call cdsapi
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c = None if dry_run else cdsapi.Client()

    for date, hours in sorted(date_groups.items()):
        y, m, d = date.split("-")

        for dataset, vars_, levels, prefix in [
            (
                "reanalysis-era5-single-levels",
                SFC_VARS,
                None,
                "era5_sfc",
            ),
            (
                "reanalysis-era5-pressure-levels",
                PLEV_VARS,
                PRESSURE_LEVELS,
                "era5_plev",
            ),
        ]:
            target = out_dir / f"{prefix}_{date}.nc"

            if target.exists():
                print(f" [skip] {target.name}")
                continue

            request = {
                "product_type": "reanalysis",
                "variable":     vars_,
                "year":  y,
                "month": m,
                "day":   d,
                "time":  hours,
                "area":  AREA,
                "data_format": "netcdf",
            }
            if levels:
                request['pressure_level'] = levels

            if dry_run:
                print(f" [dry] {target.name} hours={hours} vars={vars_}")

            else:
                print(f" requesting {target.name} ...")
                c.retrieve(dataset, request, str(target))
                print(f" -> saved {target.name}")

    # Always write data lists so they reflect current out_dir contents
    write_data_lists(date_groups, out_dir)

# convenience: derive timestamps from picks_df (for use in jupyter notebook!!)
def timestamps_from_picks(picks_df):
    """
    Extract (date, hour) pairs from the picks_df produced by Stage C.
    Deduplicates in case two tracks share a min-MSL timestamp.

    Usage in notebook
    -----------------
    from download_era5_composite import timestamps_from_picks, build_manifest, download, OUT_DIR

    timestamps  = timestamps_from_picks(picks_df)
    date_groups = build_manifest(timestamps)
    download(date_groups, OUT_DIR, dry_run=True)   # preview
    download(date_groups, OUT_DIR, dry_run=False)  # submit
    """
    times = pd.to_datetime(picks_df["datetime"], utc=True)
    return [
        (t.strftime("%Y-%m-%d"), t.strftime("%H:00"))
        for t in sorted(times.drop_duplicates())
    ]

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5 snapshots for ETC compositing."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually submit to CDS. Default is dry run."
    )        
    args = parser.parse_args()

    if not TIMESTAMPS:
        raise ValueError(
            "TIMESTAMPS list is empty.\n"
            "Either populate it above, or call timestamps_from_picks(picks_df) "
            "from your notebook instead of running this script directly."
        )
    
    date_groups = build_manifest(TIMESTAMPS)
    n_ts = sum(len(h) for h in date_groups.values())
    print(f"Dates            : {len(date_groups)}")
    print(f"Total timestamps : {n_ts}")
    print(f"Output dir       : {OUT_DIR.resolve()}\n")

    download(date_groups, OUT_DIR, dry_run=not args.download)

    





