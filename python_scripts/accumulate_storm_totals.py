"""
accumulate_storm_totals.py
==========================
Stage 3 of the ETC snowfall pipeline: sum each cyclone's masked daily fields
into a single storm-total snowfall field. 

Reads the Stage 2 output (attribute_snowfall.py) and produces one stacked
netCDF per product (era5 or nohrsc):
    
    storm_totals_{product}_cap{N}.nc
        storm_total_sf / storm_total_snowfall_depth  (track_id, latitude. longitude)
        n_days, first_spc_day, last_spc_day, status  (track_id)
        
The stack carries all track_id's from etc_summary_final.csv, not just the
attributable ones, so it joins cleanly.

ZERO vs NaNs:
-------------
0.0 = attributed, and no snow fell in that cell
NaN = NOT attributed

The 92 ETCs excluded in stage 2 (42 never within 12 deg of CONUS, 50 whose days
fall outside the daily record) are stored as all NaN slices, not zeros. Storing
them as 0s would make it did not snow = we have no data

This stage produces the base gridded product only (no statistics - that's stage
4). 

Usage
-----
    python accumulate_storm_totals.py --product era5
    python accumulate_storm_totals.py --product nohrsc
    python accumulate_storm_totals.py --product era5 --cap 10
    python accumulate_storm_totals.py --product era5 --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ETC_SUMMARY = Path("/home1/lepique/circs_research/data/etc_summary_final.csv")
OUT_ROOT = Path("/data1/lepique/etc_snowfall/")

# product -> (masked variable name, storm-total variable name, units, long_name)
PRODUCTS = {
    "era5": ("sf", "storm_total_sf", "m of water equivalent",
             "Storm-total snowfall (water equivalent, Hawcroft radial cap"),
    "nohrsc": ("snowfall_depth", "storm_total_snowfall_depth", "m",
               "Storm-total snowfall accumulation DEPTH, "
               "Hawcroft radial cap"),
}

COMPRESSION = {"zlib": True, "complevel": 4}  ## fields are mostly zeros.

def load_stage2(product, cap, out_root=OUT_ROOT):
    """Locate and load the Stage 2 index and manifest for this product/cap"""
    work = Path(out_root) / f"{product}_cap{cap:g}"
    idx_p, man_p = work / "etc_day_index.csv", work / "attribution_manifest.csv"
    for p in (idx_p, man_p):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}\n    run attribut_snowfall.py"
                     f"--product {product} --cap {cap:g} first")
    idx = pd.read_csv(idx_p, parse_dates=["spc_day", "datetime"])
    man = pd.read_csv(man_p)
    return work, idx, man

def accumulate(product="era5", cap=12, out_root=OUT_ROOT, etc_summary=ETC_SUMMARY):
    """Sum each ETC's masked days into one 2D field; return the stacked Dataset.
    
    Slices are ordered by the full track_id list from etc_summary_final.csv so
    the stack is directly indexable by cylone id, with unattributed storms
    present as NaNs.
    """
    src_var, tot_var, units, long_name = PRODUCTS[product]
    work, idx, man = load_stage2(product, cap, out_root)

    all_ids = sorted(pd.read_csv(etc_summary).track_id.astype(int))
    pos = {t: i for i, t in enumerate(all_ids)}

    with xr.open_dataset(idx.in_path.iloc[0]) as ref:
        lat, lon = ref.latitude.values, ref.longitude.values
        lon_conv = ref.attrs.get("lon_convention", "unknown")
    
    tot = np.full((len(all_ids), len(lat), len(lon)), np.nan, np.float32)
    n_days = np.zeros(len(all_ids), np.int32)
    first = np.full(len(all_ids), "", object)
    last = np.full(len(all_ids), "", object)

    md = work / "masked"
    for tid, g in idx.groupby("track_id"):
        g = g.sort_values("spc_day")
        acc = np.zeros((len(lat), len(lon)), np.float32)
        for _, r in g.iterrows():
            f = md / f"masked_{int(tid):05d}_{r.spc_day:%Y%m%d}.nc"
            if not f.exists():
                sys.exit(f"ERROR: missing mask {f}; stage 2 output is incomplete")
            with xr.open_dataset(f) as d:
                # masks are zero outside the cap by construction (from NodeFileFilter)
                # so a plain add is the way to go - no skipna needed.
                acc += d[src_var].isel(time=0).values
        i = pos[int(tid)]
        tot[i] = acc
        n_days[i] = len(g)
        first[i] = f"{g.spc_day.iloc[0]:%Y-%m-%d}"
        last[i] = f"{g.spc_day.iloc[-1]:%Y-%m-%d}"

    status = (man.set_index("track_id").status
              .reindex(all_ids).fillna("not_in_summary").values.astype(str))

    ds = xr.Dataset(
        {
            tot_var: (("track_id", "latitude", "longitude"), tot),
            "n_days": (("track_id",), n_days),
            "first_spc_day": (("track_id",), first.astype(str)),
            "last_spc_day": (("track_id",), last.astype(str)),
            "status": (("track_id",), status),
        },
        coords={"track_id": np.array(all_ids, np.int32),
                "latitude": lat, "longitude": lon},
    )

    ds[tot_var].attrs.update(
        units=units, long_name=long_name,
        _FillValue_meaning="NaN = not attributed (see status), 0.0 = attributed, no snow",
    )
    ds["n_days"].attrs["long_name"] = "SPC convective days summed into the storm total"
    ds["status"].attrs["long_name"] = "Stage 2 attribution eligibility"
    ds.attrs.update(
        attribution_method="Hawcroft et al. 2012 radial cap",
        cap_radius_gcd=float(cap),
        source={"era5": "ERA5 reanalysis-era5-single-levels, variable snowfall (sf)",
                "nohrsc": "NOHRSC National Gridded Snowfall Analysis v2, 24-h product"}[product],
        temporal_method=("12Z-12Z SPC convective-day accumulation, summed over the "
                         "cyclone lifecycle; cap anchored on the 00Z window midpoint"),
        lon_convention=lon_conv,
        overlap_note=("caps of co-existing cyclones overlap; storm totals are NOT a "
                      "partition and must not be summed across ETCs to recover a "
                      "CONUS total"),
        stage2_source=str(work),
    )
    return ds, idx

def write(ds, product, cap, out_root=OUT_ROOT):
    tot_var = PRODUCTS[product][1]
    out = Path(out_root) / f"storm_totals_{product}_cap{cap:g}.nc"
    tmp = out.with_suffix(".nc.tmp")
    enc = {tot_var: COMPRESSION, "n_days": COMPRESSION}
    ds.to_netcdf(tmp, encoding=enc, engine="netcdf4")
    tmp.replace(out)
    return out 

def summary_csv(ds, product, cap, out_root=OUT_ROOT):
    """Per ETC validation bookkeeping"""
    tot_var = PRODUCTS[product][1]
    v = ds[tot_var].values
    df = pd.DataFrame({
        "track_id": ds.track_id.values,
        "status": ds.status.values,
        "n_days": ds.n_days.values,
        "first_spc_day": ds.first_spc_day.values,
        "last_spc_day": ds.last_spc_day.values,
        "field_sum": np.nansum(v, axis=(1, 2)),
        "field_max": np.nanmax(np.where(np.isnan(v), -np.inf, v), axis=(1, 2)),
        "nonzero_cells": np.nansum(v != 0, axis=(1, 2)),
    })
    df.loc[df.n_days == 0, ["field_sum", "field_max", "nonzero_cells"]] = np.nan
    p = Path(out_root) / f"storm_totals_{product}_cap{cap:g}_summary.csv"
    df.to_csv(p, index=False)
    return p, df

def verify(ds, idx, product, cap, out_root=OUT_ROOT, n=60):
    """
    Independently re sum a random sample of ETCs from the mask files and
    compare against the stack, and confirm every storm total is boudned by
    the sum of the daily whole domain totals it drew from. (Kind of unnecesary).
    """
    src_var, tot_var = PRODUCTS[product][0], PRODUCTS[product][1]
    work = Path(out_root) / f"{product}_cap{cap:g}"
    rng = np.random.default_rng(0)
    tids = rng.choice(sorted(idx.track_id.unique()), min(n, idx.track_id.nunique()),
                      replace=False)
    worst = 0.0
    bound_viol = 0
    for tid in tids:
        g = idx[idx.track_id == tid]
        acc = None
        cap_sum = 0.0
        for _, r in g.iterrows():
            with xr.open_dataset(work / "masked" /
                                 f"masked_{int(tid):05d}_{r.spc_day:%Y%m%d}.nc") as d:
                v = d[src_var].isel(time=0).values
            acc = v.copy() if acc is None else acc + v
            with xr.open_dataset(r.in_path) as d:
                cap_sum += float(np.nansum(d[src_var].isel(time=0).values))
        got = ds[tot_var].sel(track_id=int(tid)).values
        worst = max(worst, float(np.nanmax(np.abs(got - acc))))
        if float(np.nansum(acc)) > cap_sum + 1e-6:
            bound_viol += 1
    print(f"  re-summed {len(tids)} ETCs: max |stack - independent| = {worst:.3e}")
    print(f"  storm totals exceeding their daily-domain bound: {bound_viol}")
    return worst, bound_viol

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=sorted(PRODUCTS), default="era5")
    p.add_argument("--cap", type=float, default=12.0)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--verify", action="store_true",
                   help="re-sum a sample of ETCs straight from the masks and compare")
    a = p.parse_args()

    ds, idx = accumulate(a.product, a.cap, a.out_root)
    tot_var = PRODUCTS[a.product][1]
    out = write(ds, a.product, a.cap, a.out_root)
    csv_p, df = summary_csv(ds, a.product, a.cap, a.out_root)

    att = df.n_days > 0
    print(f"product={a.product}  cap={a.cap:g} deg")
    print(f"  ETCs in stack        : {len(df)}")
    print(f"  attributed           : {int(att.sum())}")
    print(f"  NaN (not attributed) : {int((~att).sum())}  "
          + ", ".join(f"{k}={v}" for k, v in
                      df.loc[~att, 'status'].value_counts().items()))
    print(f"  days summed          : {int(df.n_days.sum()):,}")
    print(f"  field_sum  median {df.loc[att,'field_sum'].median():.2f}  "
          f"max {df.loc[att,'field_sum'].max():.2f}")
    print(f"  field_max  median {df.loc[att,'field_max'].median():.4f}  "
          f"max {df.loc[att,'field_max'].max():.4f}")
    print(f"  -> {out}  ({out.stat().st_size/1e6:.0f} MB)")
    print(f"  -> {csv_p}")

    if a.verify:
        print("\nverification:")
        verify(ds, idx, a.product, a.cap, a.out_root)


        



