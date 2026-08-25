"""
compute_derived_composite.py
============================
Compute derived meteorological variables for the ERA5 compositing dataset and
write them to per-date companion files that TempestExtremes ``NodeFileCompose``
can composite, alongside the raw ``era5_sfc_*.nc`` / ``era5_plev_*.nc`` files in
``DATA_DIR``.

For each date that has BOTH a plev and an sfc file, this writes
``era5_derived_<YYYY-MM-DD>.nc`` containing:

  thetae      equivalent potential temperature, 3-D on all 21 pressure levels
              (valid_time, pressure_level, latitude, longitude), K.
              Composite any level later with ``--var thetae(idx)``.
  shear06_u   0-6 km bulk wind shear, u-component  (valid_time, lat, lon), m/s
  shear06_v   0-6 km bulk wind shear, v-component
  shear06     0-6 km bulk wind shear, magnitude
  shear03_u   0-3 km bulk wind shear, u-component
  shear03_v   0-3 km bulk wind shear, v-component
  shear03     0-3 km bulk wind shear, magnitude

Bulk shear is height-interpolated to TRUE height above ground (AGL): plev winds
u,v are linearly interpolated -- per grid column -- to 3000 m and 6000 m AGL
using ``H_AGL = (z - z_sfc) / g`` (z = plev geopotential, z_sfc = the once-off
ERA5 orography in ``era5_orography.nc``). The near-surface base wind is the 10 m
wind (u10/v10 in the sfc files). The stored components composite correctly under
averaging; the magnitude is provided for direct plotting.

theta-e is stored 3-D so any level can be composited later without recomputing.

Coordinates (valid_time, pressure_level, latitude, longitude, and number/expver
if present) are copied verbatim from the source plev file so the derived files
read identically to the raw files under ``--lonname longitude --latname
latitude``.

Resumable and atomic: an existing ``era5_derived_*.nc`` is skipped (unless
``--overwrite``); each file is written to a ``.part`` sidecar and renamed on
success. After processing, ``in_data_list_derived.txt`` (absolute paths, one per
line, chronological) is (re)written for ``NodeFileCompose --in_data_list``.

Run in the ``storm`` conda env (has metpy, xarray, netCDF4, dask):
    STORM=/home1/lepique/miniforge3/envs/storm/bin/python
    $STORM compute_derived_composite.py                       # dry run (lists)
    $STORM compute_derived_composite.py --run --workers 4     # compute all
    $STORM compute_derived_composite.py --run --season-min 2020 --season-max 2020
"""

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")  # metpy/pint unit + interpolation chatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data1/lepique/era5_TE_composite/")
OROG_FILENAME = "era5_orography.nc"
DERIV_PREFIX = "era5_derived_"
DERIV_LIST = "in_data_list_derived.txt"

G = 9.80665                       # geopotential -> height
SHEAR_HEIGHTS = {"06": 6000.0, "03": 3000.0, "01": 1000.0}  # m AGL, top of each shear layer
SEASON_MONTHS = (12, 1, 2, 3)     # DJFM; December attributed to following year
DROP_ON_LOAD = []                 # keep number/expver so files match the raw set

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ---------------------------------------------------------------------------
# Date discovery / season filtering
# ---------------------------------------------------------------------------
def season_year(date_str):
    """DJFM season label: December belongs to the following year's JFM season."""
    y, m, _ = (int(x) for x in date_str.split("-"))
    return y + 1 if m == 12 else y


def paired_dates(data_dir):
    """Dates (YYYY-MM-DD) that have BOTH an era5_plev and era5_sfc file."""
    def dates_for(prefix):
        out = set()
        for p in glob.glob(str(Path(data_dir) / f"{prefix}_*.nc")):
            m = _DATE_RE.search(os.path.basename(p))
            if m:
                out.add(m.group(1))
        return out
    return sorted(dates_for("era5_plev") & dates_for("era5_sfc"))


def filter_dates(dates, season_min, season_max):
    if season_min is None and season_max is None:
        return dates
    out = []
    for d in dates:
        sy = season_year(d)
        if season_min is not None and sy < season_min:
            continue
        if season_max is not None and sy > season_max:
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Derived-variable math
# ---------------------------------------------------------------------------
def _interp_to_height(field, height, target):
    """
    Linearly interpolate ``field`` to ``target`` height along the pressure-level
    axis (axis=1), fully vectorized over (valid_time, lat, lon).

    field, height : arrays shaped (valid_time, level, lat, lon). ``height`` must
        increase along the level axis (true for H_AGL: 1000 hPa low, 100 hPa
        high). ``target`` : scalar height (m).

    Returns array (valid_time, lat, lon). Columns whose target lies outside the
    profile are linearly extrapolated from the nearest bracket (clamped index);
    3/6 km always sit well inside the 1000-100 hPa layer, so this only guards
    edge cases (e.g. tall terrain pushing the base level below ground).
    """
    nlev = field.shape[1]
    # number of levels below target = index of the first level >= target
    idx_hi = np.clip((height < target).sum(axis=1, keepdims=True), 1, nlev - 1)
    idx_lo = idx_hi - 1
    h_hi = np.take_along_axis(height, idx_hi, axis=1)
    h_lo = np.take_along_axis(height, idx_lo, axis=1)
    f_hi = np.take_along_axis(field, idx_hi, axis=1)
    f_lo = np.take_along_axis(field, idx_lo, axis=1)
    w = (target - h_lo) / (h_hi - h_lo)
    out = f_lo + w * (f_hi - f_lo)
    return out[:, 0]  # drop the singleton level axis


def compute_thetae(plev):
    """3-D equivalent potential temperature (K) from t, q on all levels."""

    t = plev["t"].values                      # (vt, lev, lat, lon), K
    q = plev["q"].values                      # kg/kg
    p = plev["pressure_level"].values         # hPa, (lev,)
    p3d = np.broadcast_to(p[None, :, None, None], t.shape)

    td = mpcalc.dewpoint_from_specific_humidity(
        p3d * units.hPa, t * units.K, q * units("kg/kg")
    )
    thetae = mpcalc.equivalent_potential_temperature(
        p3d * units.hPa, t * units.K, td
    )
    return np.asarray(thetae.to("K").magnitude, dtype=np.float32)


def compute_shear(plev, sfc, z_sfc):
    """
    Height-AGL bulk shear components + magnitude at each layer top in
    SHEAR_HEIGHTS, using the 10 m wind as the near-surface base.

    Returns dict of float32 arrays shaped (valid_time, lat, lon).
    """
    z = plev["z"].values                              # m^2/s^2
    u = plev["u"].values                              # m/s
    v = plev["v"].values
    h_agl = (z - z_sfc[None, None, :, :]) / G         # (vt, lev, lat, lon), m

    u10 = sfc["u10"].values                           # (vt, lat, lon)
    v10 = sfc["v10"].values

    out = {}
    for tag, htop in SHEAR_HEIGHTS.items():
        u_top = _interp_to_height(u, h_agl, htop)
        v_top = _interp_to_height(v, h_agl, htop)
        su = (u_top - u10).astype(np.float32)
        sv = (v_top - v10).astype(np.float32)
        out[f"shear{tag}_u"] = su
        out[f"shear{tag}_v"] = sv
        out[f"shear{tag}"] = np.hypot(su, sv).astype(np.float32)
    return out


def build_dataset(plev, thetae, shear):
    """Assemble the derived xarray Dataset with coords copied from ``plev``."""
    coords = {
        "valid_time": plev["valid_time"],
        "pressure_level": plev["pressure_level"],
        "latitude": plev["latitude"],
        "longitude": plev["longitude"],
    }
    for c in ("number", "expver"):
        if c in plev.coords:
            coords[c] = plev[c]

    dims3 = ("valid_time", "pressure_level", "latitude", "longitude")
    dims2 = ("valid_time", "latitude", "longitude")

    data = {
        "thetae": (dims3, thetae,
                   {"units": "K",
                    "long_name": "equivalent potential temperature"}),
    }
    labels = {
        "shear06_u": "0-6 km bulk wind shear, u-component",
        "shear06_v": "0-6 km bulk wind shear, v-component",
        "shear06":   "0-6 km bulk wind shear magnitude",
        "shear03_u": "0-3 km bulk wind shear, u-component",
        "shear03_v": "0-3 km bulk wind shear, v-component",
        "shear03":   "0-3 km bulk wind shear magnitude",
        "shear01_u": "0-1 km bulk wind shear, u-component",
        "shear01_v": "0-1 km bulk wind shear, v-component",
        "shear01":   "0-1 km bulk wind shear magnitude",
    }
    for name, arr in shear.items():
        data[name] = (dims2, arr,
                      {"units": "m s-1", "long_name": labels[name]})

    ds = xr.Dataset(data, coords=coords)
    ds.attrs["comment"] = (
        "Derived from ERA5 compositing files by compute_derived_composite.py. "
        "Bulk shear is height-AGL interpolated (base = 10 m wind); theta-e is "
        "3-D equivalent potential temperature."
    )
    return ds


# ---------------------------------------------------------------------------
# Per-date driver
# ---------------------------------------------------------------------------
def load_orography(data_dir):
    """Return z_sfc (surface geopotential, m^2/s^2) as a (lat, lon) array."""
    path = Path(data_dir) / OROG_FILENAME
    if not path.exists():
        raise SystemExit(
            f"ERROR: missing orography {path}. Download it once with:\n"
            f"  python download_era5_composite.py --orography --download"
        )
    ds = xr.open_dataset(path)
    z = ds["z"]
    for d in ("valid_time", "time", "number", "expver"):
        if d in z.dims:
            z = z.isel({d: 0})
    return np.asarray(z.values, dtype=np.float64)


def derived_path(data_dir, date):
    return Path(data_dir) / f"{DERIV_PREFIX}{date}.nc"


def process_date(date, data_dir, z_sfc, overwrite=False):
    """Compute and write era5_derived_<date>.nc. Returns a status string."""
    dst = derived_path(data_dir, date)
    if dst.exists() and not overwrite:
        return f"[skip] {dst.name}"

    plev_p = Path(data_dir) / f"era5_plev_{date}.nc"
    sfc_p = Path(data_dir) / f"era5_sfc_{date}.nc"
    plev = xr.open_dataset(plev_p)
    sfc = xr.open_dataset(sfc_p)
    try:
        if "u10" not in sfc or "v10" not in sfc:
            plev.close(); sfc.close()
            return (f"[FAIL] {dst.name}: sfc file lacks u10/v10 -- re-download "
                    f"sfc with the 10 m winds first")
        # Align sfc times to plev (same download hours; guard order/mismatch).
        sfc = sfc.reindex(valid_time=plev["valid_time"])

        thetae = compute_thetae(plev)
        shear = compute_shear(plev, sfc, z_sfc)
        ds = build_dataset(plev, thetae, shear)

        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
               for v in ds.data_vars}
        tmp = dst.with_name(dst.name + ".part")
        ds.to_netcdf(tmp, encoding=enc)
        ds.close()
        tmp.replace(dst)  # atomic: only a complete file gets the real name
        return f"[ok]   {dst.name}"
    finally:
        plev.close()
        sfc.close()


def write_data_list(dates, data_dir):
    """Write in_data_list_derived.txt: absolute paths, chronological."""
    out = Path(data_dir) / DERIV_LIST
    lines = [str(derived_path(data_dir, d).resolve()) for d in sorted(dates)]
    out.write_text("\n".join(lines) + "\n")
    print(f"  data list (derived) -> {out}  ({len(lines)} dates)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute derived ERA5 fields (theta-e, bulk shear) for "
                    "TempestExtremes compositing."
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help=f"Directory with era5_sfc/plev/orography files and "
                         f"where era5_derived_*.nc are written (default: {DATA_DIR}).")
    ap.add_argument("--run", action="store_true",
                    help="Actually compute+write. Default is a dry run that only "
                         "lists the dates that would be processed.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute derived files that already exist.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (default: 1).")
    ap.add_argument("--season-min", type=int, default=None,
                    help="Only dates whose DJFM season year is >= this "
                         "(December -> following year's season).")
    ap.add_argument("--season-max", type=int, default=None,
                    help="Only dates whose DJFM season year is <= this.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    dates = filter_dates(paired_dates(data_dir), args.season_min, args.season_max)
    print(f"Paired dates selected: {len(dates)}"
          + (f"  (season {args.season_min}-{args.season_max})"
             if (args.season_min or args.season_max) else ""))

    if not args.run:
        pending = [d for d in dates
                   if not derived_path(data_dir, d).exists() or args.overwrite]
        print(f"[dry] would process {len(pending)} date(s); "
              f"{len(dates) - len(pending)} already done.")
        for d in dates[:10]:
            state = "exists" if derived_path(data_dir, d).exists() else "TODO"
            print(f"  {d}  ({state})")
        if len(dates) > 10:
            print(f"  ... (+{len(dates) - 10} more)")
        # Still (re)write the data list so it reflects planned contents.
        write_data_list(dates, data_dir)
        return

    z_sfc = load_orography(data_dir)
    print(f"Orography loaded: elevation min/mean/max = "
          f"{(z_sfc/G).min():.0f} / {(z_sfc/G).mean():.0f} / "
          f"{(z_sfc/G).max():.0f} m")

    n_ok = n_skip = n_fail = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_date, d, data_dir, z_sfc, args.overwrite): d
                    for d in dates}
            for fut in as_completed(futs):
                msg = fut.result()
                print(" " + msg)
                n_ok += msg.startswith("[ok]")
                n_skip += msg.startswith("[skip]")
                n_fail += msg.startswith("[FAIL]")
    else:
        for d in dates:
            msg = process_date(d, data_dir, z_sfc, args.overwrite)
            print(" " + msg)
            n_ok += msg.startswith("[ok]")
            n_skip += msg.startswith("[skip]")
            n_fail += msg.startswith("[FAIL]")

    write_data_list(dates, data_dir)
    print(f"\nDONE. wrote {n_ok}, skipped {n_skip}, failed {n_fail}.")


if __name__ == "__main__":
    main()
