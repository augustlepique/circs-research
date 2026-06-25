"""
download_era5_composite.py
==========================
Download ERA5 data to feed TempestExtremes ``NodeFileCompose`` for
cyclone-relative composites.

For each cyclone (track) in ``etc_summary`` we composite around 9 reference
times: the min-MSL time and offsets of -24, -18, -12, -6, 0, +6, +12, +18,
+24 hours relative to min-MSL.  The union of every (date, hour) across all
cyclones and all offsets is deduplicated, and only the unique timestamps are
downloaded (many cyclones share timestamps, so dedup is essential).

Offset times that fall outside the track's existence are still downloaded
(they are valid ERA5 clock times); offset times that fall outside the DJFM
season window are downloaded but flagged in the manifest and end-of-run
summary.

Produces, in ``OUT_DIR``:
  era5_sfc_<YYYY-MM-DD>.nc    single-level fields, all needed hours of a date
  era5_plev_<YYYY-MM-DD>.nc   pressure-level fields, all needed hours of a date
  in_data_list_sfc.txt        absolute paths, one per line, chronological
  in_data_list_plev.txt       (consumed by NodeFileCompose --in_data_list)
  manifest.csv                every unique timestamp + how many cyclones/offsets
                              reference it, and whether it is in-season

The full multi-level download is intentional: derived parameters (MUCAPE,
SREH, EHI, lifted index, 850 theta-e, 500 hPa height/vorticity, 300/850 winds,
...) are computed later at the analysis stage with metpy, not at download time.

Usage
-----
Command line (dry run is the default; --download actually submits to CDS):
    python download_era5_composite.py                      # dry run
    python download_era5_composite.py --download           # submit to CDS
    python download_era5_composite.py --etc-summary etc_summary_2.csv \
        --out-dir /data1/lepique/era5_TE_composite/ --download

From a notebook:
    from download_era5_composite import (
        timestamps_from_etc_summary, build_date_groups, download, OUT_DIR,
    )
    manifest, expanded = timestamps_from_etc_summary(etc_summary)
    date_groups        = build_date_groups(manifest)
    download(date_groups, OUT_DIR, dry_run=True,  manifest=manifest, expanded=expanded)  # preview
    download(date_groups, OUT_DIR, dry_run=False, manifest=manifest, expanded=expanded)  # submit

CDS-Beta API notes (current requirements):
  * use cdsapi against the new CDS-Beta endpoint
  * use "data_format": "netcdf"   (NOT "format": "netcdf" -- deprecated, fails)
  * use "pressure_level" singular  (NOT "pressure_levels" -- MARS ambiguity error)
  * product_type is "reanalysis"
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path("/data1/lepique/era5_TE_composite/")

# ERA5 domain [N, W, S, E]. Generous enough for a ~20 deg composite radius
# around cyclones as far south as ~27 N.
AREA = [70, -140, -10, -50]

# Reference-time offsets relative to each cyclone's min-MSL time (hours).
OFFSET_HOURS = [-24, -18, -12, -6, 0, 6, 12, 18, 24]

# DJFM season window. Offset times whose month is not in this set are still
# downloaded but flagged (e.g. an early-December -24h spilling into November,
# or a late-March +24h spilling into April).
SEASON_MONTHS = (12, 1, 2, 3)

# etc_summary column names
COL_TRACKID = "track_id"
COL_TIMEMSL = "time_min_msl"

# Single-level fields (reanalysis-era5-single-levels)
SFC_VARS = [
    "mean_sea_level_pressure",                 # -> msl   (Pa)
    "convective_available_potential_energy",   # -> cape  (J kg-1) ERA5 native ref
    "2m_temperature",                          # -> t2m   (K)
    "2m_dewpoint_temperature",                 # -> d2m   (K)
    "surface_pressure",                        # -> sp    (Pa)
    "total_column_water_vapour",               # -> tcwv  (kg m-2)
]

# Pressure-level fields (reanalysis-era5-pressure-levels), full troposphere
PLEV_VARS = [
    "temperature",            # -> t   (K)
    "specific_humidity",      # -> q   (kg kg-1)
    "relative_humidity",      # -> r   (%)
    "u_component_of_wind",    # -> u   (m s-1)
    "v_component_of_wind",    # -> v   (m s-1)
    "geopotential",           # -> z   (m2 s-2)
    "vorticity",              # -> vo  (s-1) relative vorticity
]

PRESSURE_LEVELS = [
    "1000", "975", "950", "925", "900", "850", "800", "750", "700", "650",
    "600", "550", "500", "450", "400", "350", "300", "250", "200", "150", "100",
]


# ---------------------------------------------------------------------------
# Timestamp expansion
# ---------------------------------------------------------------------------
def timestamps_from_etc_summary(
    etc_summary,
    time_col=COL_TIMEMSL,
    track_col=COL_TRACKID,
    offsets=OFFSET_HOURS,
    season_months=SEASON_MONTHS,
):
    """
    Expand each cyclone's reference time by ``offsets`` and deduplicate.

    All arithmetic is done on tz-aware (UTC) timestamps with timedeltas, so
    rollover into adjacent months/years is handled correctly.

    Parameters
    ----------
    etc_summary : DataFrame with ``track_col`` and ``time_col`` (the latter
        parseable as UTC datetimes, e.g. the ``time_min_msl`` column).
    offsets : iterable of int hours relative to each reference time.
    season_months : months considered "in-season" (DJFM by default).

    Returns
    -------
    manifest : DataFrame, one row per unique (date, hour):
        date, hour, n_refs (cyclone-offset pairs hitting it),
        n_cyclones (distinct tracks), in_season (bool).
    expanded : DataFrame, one row per (cyclone, offset):
        track_id, offset_h, timestamp, date, hour, in_season.
    """
    es = etc_summary.dropna(subset=[time_col]).copy()
    base = pd.to_datetime(es[time_col], utc=True)

    records = []
    for tid, t0 in zip(es[track_col], base):
        for off in offsets:
            ts = t0 + pd.Timedelta(hours=off)
            records.append({"track_id": tid, "offset_h": int(off), "timestamp": ts})

    expanded = pd.DataFrame(records)
    expanded["in_season"] = expanded["timestamp"].dt.month.isin(season_months)
    expanded["date"] = expanded["timestamp"].dt.strftime("%Y-%m-%d")
    expanded["hour"] = expanded["timestamp"].dt.strftime("%H:00")

    manifest = (
        expanded.groupby(["date", "hour"])
        .agg(
            n_refs=("track_id", "size"),
            n_cyclones=("track_id", "nunique"),
            in_season=("in_season", "first"),
        )
        .reset_index()
        .sort_values(["date", "hour"])
        .reset_index(drop=True)
    )
    return manifest, expanded


def build_date_groups(manifest):
    """
    manifest : DataFrame with 'date' and 'hour' columns.
    Returns : dict  date_str -> sorted list of unique hour strings.
    Same-date hours are batched so each date is one CDS request per dataset.
    """
    return (
        manifest.groupby("date")["hour"]
        .apply(lambda h: sorted(h.unique().tolist()))
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Output bookkeeping
# ---------------------------------------------------------------------------
def write_data_lists(date_groups, out_dir):
    """
    Write the two NodeFileCompose --in_data_list files: absolute paths, one per
    line, chronologically sorted (one file per date per dataset).
    """
    out_dir = Path(out_dir)
    dates = sorted(date_groups)  # YYYY-MM-DD sorts chronologically

    sfc_list = out_dir / "in_data_list_sfc.txt"
    plev_list = out_dir / "in_data_list_plev.txt"

    sfc_list.write_text(
        "\n".join(str((out_dir / f"era5_sfc_{d}.nc").resolve()) for d in dates) + "\n"
    )
    plev_list.write_text(
        "\n".join(str((out_dir / f"era5_plev_{d}.nc").resolve()) for d in dates) + "\n"
    )

    print(f"  data list (sfc)  -> {sfc_list}")
    print(f"  data list (plev) -> {plev_list}")
    return sfc_list, plev_list


def write_manifest_csv(manifest, out_dir):
    """Write the unique-timestamp manifest CSV for record-keeping."""
    out_dir = Path(out_dir)
    path = out_dir / "manifest.csv"
    manifest.to_csv(path, index=False)
    print(f"  manifest         -> {path}")
    return path


def print_summary(manifest, expanded, date_groups):
    """End-of-run summary: counts, estimated requests, out-of-season offsets."""
    n_ts = len(manifest)
    n_dates = len(date_groups)
    out = expanded[~expanded["in_season"]]

    print("\n===== summary =====")
    print(f"  unique timestamps     : {n_ts}")
    print(f"  unique dates          : {n_dates}")
    print(f"  estimated CDS requests: {2 * n_dates}  (sfc + plev per date)")
    print(f"  cyclones expanded     : {expanded['track_id'].nunique()}")
    print(f"  cyclone-offset refs   : {len(expanded)}")

    if len(out):
        n_out_ts = out[["date", "hour"]].drop_duplicates().shape[0]
        print(
            f"\n  OUT-OF-SEASON: {len(out)} offset(s) across {n_out_ts} timestamp(s) "
            f"fall outside DJFM {SEASON_MONTHS} (downloaded anyway):"
        )
        show = (
            out.sort_values("timestamp")
            .groupby(["date", "hour"])
            .agg(n_refs=("track_id", "size"),
                 offsets=("offset_h", lambda s: sorted(set(s))))
            .reset_index()
        )
        for _, r in show.iterrows():
            print(f"    {r['date']} {r['hour']}  refs={r['n_refs']}  offsets={r['offsets']}")
    else:
        print(f"\n  all timestamps fall within DJFM {SEASON_MONTHS}.")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(date_groups, out_dir=OUT_DIR, dry_run=True, manifest=None, expanded=None):
    """
    Submit CDS requests for every date in ``date_groups``.

    Idempotent: any output file that already exists is skipped, so the script
    is safe to re-run after an interruption (CDS requests often queue/time out).
    Data lists, and the manifest CSV when provided, are always (re)written -- in
    dry-run mode too -- so they reflect the planned/current contents of out_dir.

    Parameters
    ----------
    date_groups : dict  date_str -> [hour_str, ...]
    out_dir     : Path
    dry_run     : if True, print planned requests but do not call cdsapi.
    manifest    : optional manifest DataFrame; if given, written to manifest.csv.
    expanded    : optional expanded DataFrame; if given, used for the summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not dry_run:
        import cdsapi  # lazy: keep the module importable without cdsapi installed
        client = cdsapi.Client()

    datasets = [
        ("reanalysis-era5-single-levels", SFC_VARS, None, "era5_sfc"),
        ("reanalysis-era5-pressure-levels", PLEV_VARS, PRESSURE_LEVELS, "era5_plev"),
    ]

    for date, hours in sorted(date_groups.items()):
        y, m, d = date.split("-")
        for dataset, vars_, levels, prefix in datasets:
            target = out_dir / f"{prefix}_{date}.nc"
            if target.exists():
                print(f" [skip] {target.name}")
                continue

            request = {
                "product_type": "reanalysis",
                "variable": vars_,
                "year": y,
                "month": m,
                "day": d,
                "time": hours,
                "area": AREA,
                "data_format": "netcdf",
            }
            if levels:
                request["pressure_level"] = levels  # singular key (required)

            if dry_run:
                lv = f" levels={len(levels)}" if levels else ""
                print(f" [dry] {target.name} hours={hours} vars={len(vars_)}{lv}")
            else:
                print(f" requesting {target.name} ...")
                client.retrieve(dataset, request, str(target))
                print(f" -> saved {target.name}")

    # Always refresh bookkeeping so it reflects current/planned out_dir contents.
    write_data_lists(date_groups, out_dir)
    if manifest is not None:
        write_manifest_csv(manifest, out_dir)
    if manifest is not None and expanded is not None:
        print_summary(manifest, expanded, date_groups)


def run(etc_summary, out_dir=OUT_DIR, dry_run=True, **kwargs):
    """
    Convenience one-call pipeline: expand timestamps from ``etc_summary``,
    build date groups, and download (or preview). Extra kwargs are forwarded to
    ``timestamps_from_etc_summary`` (e.g. ``time_col``, ``offsets``).

    Returns (manifest, expanded, date_groups).
    """
    manifest, expanded = timestamps_from_etc_summary(etc_summary, **kwargs)
    date_groups = build_date_groups(manifest)
    download(date_groups, out_dir, dry_run=dry_run, manifest=manifest, expanded=expanded)
    return manifest, expanded, date_groups


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5 reference-time data for ETC compositing."
    )
    parser.add_argument(
        "--etc-summary",
        default="etc_summary_2.csv",
        help="CSV with 'track_id' and 'time_min_msl' columns "
             "(default: etc_summary_2.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually submit to CDS. Default is a dry run.",
    )
    args = parser.parse_args()

    etc_summary = pd.read_csv(args.etc_summary)
    run(etc_summary, out_dir=Path(args.out_dir), dry_run=not args.download)
