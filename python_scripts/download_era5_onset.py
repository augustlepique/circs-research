"""
download_era5_onset.py
======================
Download the ERA5 hours needed for outbreak-cyclone composites lagged relative
to the first KDE core's onset / peak time (0, 6, 12, 18, 24 h before).

The hours come from ``needed_times.csv``, written by the onset/peak section of
TE_compositing_development.ipynb. ALL of them are downloaded to their own
directory -- including hours that also exist in era5_TE_composite -- on the
domain set by ``AREA`` in download_era5_composite.py ([80, -180, -5, -20]).
That domain covers the full +/-25 deg plotted composite window for ~96% of
composited centers; the original era5_TE_composite files use the smaller
[70, -140, -10, -50], so mixing the two would give each composite edges built
from a mix of domains. The composites therefore read ONLY this directory's
data lists (in_data_list_{sfc,plev,derived}.txt).

The orography is downloaded fresh on the same AREA (compute_derived_composite.py
needs z_sfc on the same grid as the plev/sfc files for height-AGL shear). The
first real download writes AREA.txt; later runs refuse to add files if AREA has
changed since, so the directory can never hold a mix of domains.

Steps (run from python_scripts/, era5 env for the download, storm for derived):
    python download_era5_onset.py                         # dry run: counts + requests
    python download_era5_onset.py --download --workers 3  # submit to CDS (sfc + plev + orography)
    $STORM compute_derived_composite.py --data-dir /data1/lepique/era5_TE_composite_onset --run --workers 4

DISCLAIMER: Script made with help from claude code
"""

import argparse
from pathlib import Path

import pandas as pd

import download_era5_composite as dec
from download_era5_composite import build_date_groups, download, download_orography

NEW_DIR = Path("/data1/lepique/era5_TE_composite_onset")
NEEDED = Path(__file__).resolve().parent.parent / "data" / "onset_peak_oc" / "needed_times.csv"


def needed_hours(needed_csv):
    """Unique needed hours (UTC), sorted."""
    need = (pd.to_datetime(pd.read_csv(needed_csv)["datetime"], utc=True)
            .drop_duplicates().sort_values())
    print(f"needed hours: {len(need)} on {need.dt.date.nunique()} dates   AREA [N, W, S, E] = {dec.AREA}")
    return need


def check_area(out_dir, area=dec.AREA, record=False):
    """Refuse to add to a directory downloaded on another AREA: download() skips
    existing date files, so a changed AREA would otherwise silently leave a mix
    of domains. The AREA of a directory is recorded in AREA.txt on its first
    real download (the era5 env has no netCDF reader to inspect the files)."""
    out_dir = Path(out_dir)
    marker = out_dir / "AREA.txt"
    has_data = any(out_dir.glob("era5_*_*.nc"))
    if marker.exists():
        got = marker.read_text().strip()
        if got != str(list(area)):
            raise SystemExit(f"{out_dir} was downloaded on AREA {got}, but AREA is now {list(area)}: "
                             f"remove it (or use another --out-dir) before downloading")
    elif has_data:
        raise SystemExit(f"{out_dir} has ERA5 files but no AREA.txt, so their domain is unknown: "
                         f"remove it (or use another --out-dir) before downloading")
    elif record:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{list(area)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--needed", default=str(NEEDED), help=f"needed_times.csv (default: {NEEDED})")
    ap.add_argument("--out-dir", default=str(NEW_DIR), help=f"output directory (default: {NEW_DIR})")
    ap.add_argument("--download", action="store_true", help="submit to CDS (default: dry run)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent CDS requests (3-4 is safe)")
    ap.add_argument("--only", choices=["both", "sfc", "plev"], default="both")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    check_area(out_dir, record=args.download)
    need = needed_hours(args.needed)
    manifest = pd.DataFrame({"date": need.dt.strftime("%Y-%m-%d"), "hour": need.dt.strftime("%H:00")})
    download(build_date_groups(manifest), out_dir, dry_run=not args.download,
             manifest=manifest, workers=args.workers, only=args.only)
    download_orography(out_dir=out_dir, dry_run=not args.download)
