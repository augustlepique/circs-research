"""
compute_irrotational.py
=======================
Computes the irrotational (divergent) wind with windspharm and the
advection of PV by it, and writes per-date companion files that TempestExtremes
''NodeFileCompose'' can composite alongside the raw ERA5 files in ''DATA_DIR''.

Inputs, per date:
    era5_uv_global_<date>.nc (in UV_DIR) GLOBAL 0.5 deg u, v at 300/250/200 hPa
    era5_pv_<date>.nc    (in DATA_DIR) regional 0.25 deg PV at the 21 plevs


Method (all per time step before compositing, can't calculate advection after)
    1. windspharm splits the global wind into rotational and irrotational 
    components. 
    2. Bilinear interpolation of u_chi, v_chi, and chi onto the regional
    PV grid(V_chi is smooth so little is lost from the interpolation)
    3. pvadv_chi = -(u_chi dPV/dx + v_chi dPV/dy).
    4. Pressure weighted 300-200 hPa layer means. The layer of advection is the mean of the 
    per level advections, not the advection of teh layer means.

Writes ''era5_irrot_<YYYY-MM-DD>.nc'' in DATA_DIR containing:

    uchi, vchi    irrotational components (m/s)  3D
    chi           velocity potential (m2/s)      3D
    pvadv_chi      -V_chi dot gradient(PV)       3D
    uchi_l, vchi_l, pv_l, pvadv_chi_l   300-200 hPa layer means, 2-D

3D fields hold only levels, so their level index is different from the plev map:
    idx 0 1 2 
    hPa 300 250 200         (e.g. --var "pvadv_chi(1) is 250 hPa)

PV units are in PVU (era5 native). 
Coordinates (valid_time, latitude, longitude, number/expver) are copied from 
the PV file, so the output reads identically. Resumable and atomic like the 
other scripts (e.g. compute_derived_composite.py). A full period run writes 
''in_data_list_irrot.txt'' listing every PV date, so a NodeFileCompose 
run fails loudly rather than silently compositing a subset if some dates are 
missing. 

Run in the storm conda env
    python compute_irrotational.py                                 # dry run
    $python compute_irrotational.py --run --workers 4               # min-MSL set
    $python compute_irrotational.py --run --season-min 2020 --season-max 2020
    $python compute_irrotational.py --data-dir /data1/lepique/era5_TE_composite_onset --run
"""

import argparse 
import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind

from compute_derived_composite import _DATE_RE, filter_dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data1/lepique/era5_TE_composite")      # era5 pv in, era5 irrot out
UV_DIR = Path("/data1/lepique/era5_TE_composite")        # global era5 uv files
PV_PREFIX = "era5_pv_"
UV_PREFIX = "era5_uv_global_"
IRROT_PREFIX = "era5_irrot_"
IRROT_LIST = "in_data_list_irrot.txt"

LEVELS = [300.0, 250.0, 200.0]       # hPa
A_EARTH = 6.371229e6       # m

# ---------------------------------------------------------------------------
# Date Discovery
# ---------------------------------------------------------------------------
def dates_for(data_dir, prefix):
    out = set()
    for p in glob.glob(str(Path(data_dir) / f"{prefix}*.nc")):
        m = _DATE_RE.search(os.path.basename(p))
        if m:
            out.add(m.group(1))
    return out 

# ---------------------------------------------------------------------------
# Irrotational Wind (global)
# ---------------------------------------------------------------------------
def irrotational_wind(u, v):
    """
    Irrotational wind and velocity potential from the global wind.

    u, v : DataArrays on a global grid. windspharm finds the lat/lon 
    dimensions itself, flips lat to north-south if needed, and handles 
    any extra dimensions. 

    Returns (uchi, vchi chi) DataArrays with u's dims and coords.
    """
    # legfunc = "compouted" recomputes the legendre functions on each call
    # instead of storing them 
    w = VectorWind(u, v, rsphere=A_EARTH, legfunc="computed")
    uchi, vchi = w.irrotationalcomponent()
    chi = w.velocitypotential()
    return uchi, vchi, chi

def to_regional(da, target):
    """
    bilinearly interpolate a global 0-360 field onto ''target'''s lat/lon"""
    da = da.assign_coords(longitude=((da["longitude"] + 180) % 360) - 180).sortby("longitude")
    out = da.interp(latitude=target["latitude"], longitude=target["longitude"])
    if out.isnull().any():
        raise ValueError("regional grid falls outside the global grid (aka you messed up")
    return out.values

# ---------------------------------------------------------------------------
# PV advection (regional)
# ---------------------------------------------------------------------------
def spherical_gradient(f, lat, lon):
    phi = np.deg2rad(lat)
    dfdlam = np.gradient(f, np.deg2rad(lon), axis=-1)  # longitude, np.gradient uses centered differencing
    dfdphi = np.gradient(f, phi, axis=-2) # latitude
    return dfdlam / (A_EARTH * np.cos(phi)[:,None]), dfdphi / A_EARTH


def layer_mean(x, p):
    """pressure-weighted layer mean over level axis"""
    return np.trapezoid(x, p, axis=-3) / (p[-1] - p[0])

def build_dataset(pv, uchi, vchi, chi, pvadv):
    """Assemble the output dataset with coords copied from original PV file"""
    p = np.asarray(LEVELS)
    coords = {
        "valid_time": pv["valid_time"],
        "pressure_level": pv["pressure_level"],
        "latitude": pv["latitude"],
        "longitude": pv["longitude"],
    }
    for c in ("number", "expver"):
        if c in pv.coords:
            coords[c] = pv[c]

    dims3 = ("valid_time", "pressure_level", "latitude", "longitude")
    dims2 = ("valid_time", "latitude", "longitude")
    layer = "300-200 hPa pressure weighted mean"
    ds = xr.Dataset(
        {
            "uchi": (dims3, uchi, {"units": "m s-1", "long_name": "irrotational wind, u-component"}),
            "vchi": (dims3, vchi, {"units": "m s-1", "long_name": "irrotational wind, v-component"}),
            "chi": (dims3, chi, {"units": "m2 s-1", "long_name": "velocity potential"}),
            "pvadv_chi": (dims3, pvadv, {"units": "K m2 kg-1 s-2",
                                         "long_name": "advection of PV by the irrotational wind"}),
            "uchi_l": (dims2, layer_mean(uchi, p), {"units": "m s-1",
                                                    "long_name": layer + "irrotational u"}),
            "vchi_l": (dims2, layer_mean(vchi, p), {"units": "m s-1",
                                                    "long_name": layer + "irrotational v"}),
            "pv_l": (dims2, layer_mean(pv.values, p), {"units": "K m2 kg-1 s-1",
                                                       "long_name": layer + "PV"}),
            "pvadv_chi_l": (dims2, layer_mean(pvadv, p), {"units": "K m2 kg-1 s-2",
                                                          "long_name": layer + "PV advection by v_chi"}),
        },
        coords = coords,
    )

    ds.attrs["comment"] = (
        "Irrotational wind from windspharm VectorWind(u, v) on global 0.5 deg "
        "ERA5 winds, interpolated to the PV grid; PV advection by it computed "
        "per time step. Written by compute_irrotational.py"
    )
    return ds

# ---------------------------------------------------------------------------
# per date driver
# ---------------------------------------------------------------------------
def irrot_path(data_dir, date):
    return Path(data_dir) / f"{IRROT_PREFIX}{date}.nc"


def process_date(date, data_dir, uv_dir, overwrite=False, pv_prefix = PV_PREFIX):
    """Compute and write era5_irrot_<date>.nc. Returns a status string."""
    dst = irrot_path(data_dir, date)
    if dst.exists() and not overwrite:
        return f"[skip] {dst.name}"
    uv_p = Path(uv_dir) / f"{UV_PREFIX}{date}.nc"
    if not uv_p.exists():
        return f"[wait] {dst.name}: {uv_p.name} not downloaded yet"

    pvf = xr.open_dataset(Path(data_dir) / f"{pv_prefix}{date}.nc")
    uvf = xr.open_dataset(uv_p)

    try:
        missing = np.setdiff1d(pvf["valid_time"].values, uvf["valid_time"].values)
        if missing.size:
            return (f"[FAIL] {dst.name}: {uv_p.name} lacks {missing.size} PV hour(s) "
                    f"-- re-download that date with --overwrite")
        pv = pvf["pv"].sel(pressure_level=LEVELS).load()
        # match the PV hours, drop nondimensional coords (number, expver) that
        # windspharm doesn't want
        sel = {"valid_time": pvf["valid_time"], "pressure_level": LEVELS}
        u = uvf["u"].sel(sel).reset_coords(drop=True).load()
        v = uvf["v"].sel(sel).reset_coords(drop=True).load()

        uchi, vchi, chi = (to_regional(x, pv) for x in irrotational_wind(u, v))
        dpvdx, dpvdy = spherical_gradient(pv.values.astype(np.float64),
                                          pv["latitude"].values, pv["longitude"].values)   # calculate gradient

        pvadv = -(uchi * dpvdx + vchi * dpvdy)    # advection of PV by irrotational wind

        ds = build_dataset(pv, uchi, vchi, chi, pvadv)
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
               for v in ds.data_vars}
        tmp = dst.with_name(dst.name + ".part")
        ds.to_netcdf(tmp, encoding=enc)
        tmp.replace(dst)   # atomic
        return f"[ok]  {dst.name}"
    
    except Exception as e:
        return f"[FAIL] {dst.name}: {e}"
    
    finally:
        pvf.close()
        uvf.close()


def write_data_list(dates, data_dir):
    """Write in_data_list_irrot.txt: absolute paths and chronological"""
    out = Path(data_dir) / IRROT_LIST
    lines = [str(irrot_path(data_dir, d).resolve()) for d in sorted(dates)]
    out.write_text("\n".join(lines) + "\n")
    print(f"  data list (irrot) -> {out}  ({len(lines)} dates)")
    return out 

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute the irrotational wind (windspharm) and its PV "
        "advection for TempestExtremes Compositing"
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help=f"Directory with era5_pv_*.nc, where era5_irrot_*.nc are "
                         f"written (default: {DATA_DIR}).")
    ap.add_argument("--uv-dir", default=str(UV_DIR),
                    help=f"Directory with the global era5_uv_global_*.nc (default: {UV_DIR}).")
    ap.add_argument("--run", action="store_true",
                    help="Actually compute+write. Default is a dry run.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute files that already exist.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (default: 1).")
    ap.add_argument("--season-min", type=int, default=None,
                    help="Only dates whose DJFM season year is >= this.")
    ap.add_argument("--season-max", type=int, default=None,
                    help="Only dates whose DJFM season year is <= this.")
    ap.add_argument("--pv-prefix", default=PV_PREFIX,
                    help=f"Prefix of the files folding 'pv' (default: {PV_PREFIX})."
                    f"Use era5_plev_ where PV is inside the plev files, "
                    f"e.g. the onset directory.")
    
    args = ap.parse_args()

    data_dir, uv_dir = Path(args.data_dir), Path(args.uv_dir)
    season_filtered = args.season_min is not None or args.season_max is not None
    dates = filter_dates(sorted(dates_for(data_dir, args.pv_prefix)),
                         args.season_min, args.season_max)
    if not dates:
        raise SystemExit(f"No {args.pv_prefix}*.nc files in {data_dir} -- nothing to do")
    n_uv = len(set(dates) & dates_for(uv_dir, UV_PREFIX))
    print(f"PV dates selected: {len(dates)}  ({n_uv} with global winds downloaded)")


    if not args.run:
        n_done = sum(irrot_path(data_dir, d).exists() for d in dates)
        print(f"[dry] {n_done} already done, {n_uv - n_done} ready to compute, "
              f"{len(dates) - n_uv} waiting on global winds.")
        return

    n = {"ok": 0, "skip": 0, "wait": 0, "FAIL": 0}
    def report(msg):
        print(" " + msg)
        n[msg[1:msg.index("]")]] += 1

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_date, d, data_dir, uv_dir, args.overwrite, args.pv_prefix)
                    for d in dates]
            for fut in as_completed(futs):
                report(fut.result())
    else:
        for d in dates:
            report(process_date(d, data_dir, uv_dir, args.overwrite, args.pv_prefix))

    if season_filtered:
        print(f"  [data list skipped: season-filtered run] {IRROT_LIST} left untouched.")
    else:
        write_data_list(dates, data_dir)
    print(f"\nDONE. wrote {n['ok']}, skipped {n['skip']}, waiting {n['wait']}, "
          f"failed {n['FAIL']}.")

if __name__ == "__main__":
    main()
    








