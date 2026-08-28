"""
attribute_snowfall.py
=====================
Stage 2 of the ETC snowfall pipeline: attribute daily snowfall to each tracked cyclone
via the Hawcroft et al. (2012) fixed radius. 

For every ETC, for every SPC convective day it spans, mask the daily snowfall field 
to within CAP_RADIUS (12 degrees) great circle degrees of that day's cyclone position,
using TempestExtremes NodeFileFilter. Stage 3 then sums the masked days into a 
per-ETC storm total. 

ANCHOR TIME - 0Z
----------------
The daily fields (build_daily_sf.py, download_nohrsc.py) cover the SPC 
convective day (12Z(D) - 12Z(D+1), and are stamped at 0Z(D+1), the window 
midpoint. NodeFileFilter matches node time to the data time exactly, so the 
nodefile must carry 0Z nodes. Anchoring at the midpoint keeps the cap centered
on the storm.

BATCHING
--------
One NodeFileFilter call per ETC: a single nodefile holding all that storm's 00Z
nodes, plus --in_data_list/--out_data_list over its days. TE matches each node
to its day. 

GAPS IN TRACKS
--------------
StitchNodes stitches across dropouts so many tracks have non hourly gaps, including
00Z instants. Because of that, we need to interpolate for the cyclone position at
00Z for nodes where 00Z is missing. 

NODES OUTSIDE THE GRID
----------------------
NodeFileFilter silently drops nodes outside the data grid (without any error or warning).
The daily grid is CONUS buffered by 12 degree so that any nodes whose cap reaches 
CONUS is inside it. Nodes outside are verified not to reach CONUS and are 
excluded here. 

Usage
-----
    python attribute_snowfall.py --product era5 --dry-run
    python attribute_snowfall.py --product era5
    python attribute snowfall.py --product nohrsc
    python attribute_snowfall.py --product era5 --cap 10    # a sensitivity sweep
"""

import argparse
import sys 
import subprocess
from pathlib import Path 

import numpy as np
import pandas as pd
import xarray as xr

from TE_processing import parse_nodefile 

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKS = Path("/data1/lepique/TempestExtremes/tracks/tracks_hourly_1995_2025.txt")
ETC_SUMMARY = Path("/home1/lepique/circs_research/data/etc_summary_final.csv")
OUT_ROOT = Path("/data1/lepique/etc_snowfall/")
NODEFILEFILTER = Path("/home1/lepique/miniforge3/envs/TE/bin/NodeFileFilter")

# Hawcroft et al. (2012) cap radius, GREAT-CIRCLE DEGREES (verified: kept cells
# span 0.06-12.05 deg for --bydist 12.0). Config constant so the 10/12/14 deg
# sensitivity sweep is a one-line change; outputs are written under a
# cap-labelled directory so radii never collide.
CAP_RADIUS = 12.0

# Products: key -> (daily dir, filename template, variable name)
PRODUCTS = {
    "era5":   (Path("/data1/lepique/era5_sf_daily/"),
               "daily_sf_{day:%Y%m%d}.nc", "sf"),
    "nohrsc": (Path("/data1/lepique/nohrsc_daily/"),
               "daily_snow_nohrsc_{day:%Y%m%d}.nc", "snowfall_depth"),
}

# Recover 00Z positions that fall inside a track gap by linear interpolation.
INTERPOLATE_MISSING_00Z = True

IN_FMT = "lon,lat,msl,phis"   # columns after the leading i,j pair
LATNAME, LONNAME = "latitude", "longitude"

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def grid_bounds(daily_dir, tmpl):
    """
    (lon_min, lon_max, lat_min, lat_max) read from an actual daily file.
    
    Derived rather than hardcoded.
    """
    sample = sorted(Path(daily_dir).glob(tmpl.split("{")[0] + "*.nc"))
    if not sample:
        sys.exit(f"ERROR: no daily files in {daily_dir}")
    with xr.open_dataset(sample[0]) as ds:
        lon, lat = ds[LONNAME].values,  ds[LATNAME].values
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def nodes_00z(track, interpolate=INTERPOLATE_MISSING_00Z):
    """
    The 00Z nodes of one track, optionally filling missing gap instances.

    'track' is one track's rows, sorted by datetime. REturns a DataFrame with
    datetime/lon/lat/msl/z and an 'interpolated' flag.
    
    Longitude is unwrapped before interpolating so a track crossing the 0/360 line
    does not produce a midpoint on the opposite side of the globe.
    """
    t = track.sort_values("datetime")
    got = t[t.hour == 0].copy()
    got["interpolated"] = False
    if not interpolate:
        return got

    span = pd.date_range(t.datetime.min().ceil("D"), t.datetime.max().floor("D"),
                         freq="D")
    missing = [ts for ts in span if ts not in set(got.datetime)]
    if not missing:
        return got

    x = t.datetime.astype("int64").values
    lon_unwrapped = np.degrees(np.unwrap(np.radians(t.lon.values)))
    filled = []
    for ts in missing:
        xi = np.int64(ts.value)
        if not (x.min() < xi < x.max()):
            continue                      # no bracketing pair - cannot interpolate
        filled.append({
            "track_id": t.track_id.iloc[0],
            "datetime": ts,
            "lon": float(np.interp(xi, x, lon_unwrapped)) % 360,
            "lat": float(np.interp(xi, x, t.lat.values)),
            "msl": float(np.interp(xi, x, t.msl.values)),
            "z": float(np.interp(xi, x, t.z.values)),
            "interpolated": True,
        })
    if not filled:
        return got 
    return (pd.concat([got, pd.DataFrame(filled)], ignore_index=True)
            .sort_values("datetime").reset_index(drop=True))

def build_manifest(product, tracks=TRACKS, etc_summary = ETC_SUMMARY,
                   interpolate=INTERPOLATE_MISSING_00Z):
    """
    One row per (ETC, SPC day) that can actually be attributed, plus a per-ETC
    status table explaining every ETC that produced no rows.
    
    An SPC day D is attributable when the track has (or can interpolate) a 00Z 
    node at d+1 00Z, that node is inside the data grid, and the daily file for D exists.
    """
    daily_dir, tmpl, _ = PRODUCTS[product]
    lon0, lon1, lat0, lat1 = grid_bounds(daily_dir, tmpl)

    df = parse_nodefile(tracks)
    keep = set(pd.read_csv(etc_summary).track_id.astype(int))
    df = df[df.track_id.isin(keep)]

    rows, status = [], []
    for tid, g in df.groupby("track_id"):
        n=nodes_00z(g, interpolate)
        if len(n) == 0:
            status.append((tid, "no_00z_node", 0, 0))
            continue 
        n = n.copy()
        n["spc_day"] = n.datetime - pd.Timedelta(hours=24)
        in_grid = n.lon.between(lon0, lon1) &n.lat.between(lat0, lat1)
        n=n[in_grid]
        if len(n) == 0:
            status.append((tid, "no_node_in_grid", 0, 0))
            continue
        n["path"] = [Path(daily_dir) / tmpl.format(day=d) for d in n.spc_day]
        n = n[[p.exists() for p in n.path]]
        if len(n) == 0:
            status.append((tid, "no_daily_file", 0, 0))
            continue
        status.append((tid, "ok", len(n), int(n.interpolated.sum())))
        for _, r in n.iterrows():
            rows.append({"track_id": tid, "spc_day": r.spc_day, "datetime": r.datetime,
                         "lon": r.lon, "lat": r.lat, "msl": r.msl, "z": r.z,
                         "interpolated": r.interpolated, "in_path": r.path})

    man = pd.DataFrame(rows)
    stat = pd.DataFrame(status, columns = ["track_id", "status", "n_days", "n_interpolated"])
    return man, stat 



# ---------------------------------------------------------------------------
# NodeFileFilter
# ---------------------------------------------------------------------------
def write_nodefile(path, g):
    """
    StitchNodes-format nodefile holding one ETC's 00Z nodes.
    
    The leading i,j columns are written as zeros: NodeFileFilter locates nodes
    from the lat/lon in --in_fmt and never reads them. msl is restored to Pa,
    since parse_nodefile converts it to hPa when it reads it.
    """
    g = g.sort_values("datetime")
    with open(path, "w") as f:
        h = g.iloc[0].datetime
        f.write(f"start\t{len(g)}\t{h.year}\t{h.month}\t{h.day}\t{h.hour}\n")
        for _, r in g.iterrows():
            d = r.datetime
            f.write(f"\t0\t0\t{r.lon:.6f}\t{r.lat:.6f}\t{r.msl * 100:.2f}\t{r.z:.4f}\t"
                    f"{d.year}\t{d.month}\t{d.day}\t{d.hour}\n")


def run_etc(tid, g, var, cap, work, overwrite=False):
    """
    One NodeFileFilter call covering an entire ETC.
    
    Returns (status, n_written). Idempotent.
    
    NOTE: --fillvalue is deliberately not passed. Passing it sets outside-cap cells to NaNs,
    but we want 0s for summing it later.
    """
    nd, md = work / "nodefiles", work / "masked"
    nd.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)

    outs = [md / f"masked_{tid:05d}_{d:%Y%m%d}.nc" for d in g.spc_day]
    if not overwrite and all(p.exists() for p in outs):
        return "cached", len(outs)

    #nf = nd / f"etc_{tid:05d}.txt", nd / f"out_{tid:05d}.txt"
    nf = nd / f"etc_{tid:05d}.txt"
    write_nodefile(nf,g)
    in_list, out_list = nd / f"in_{tid:05d}.txt", nd / f"out_{tid:05d}.txt"
    in_list.write_text("\n".join(str(p) for p in g.in_path) + "\n")
    out_list.write_text("\n".join(str(p) for p in outs) + "\n")

    cmd = [str(NODEFILEFILTER),
           "--in_nodefile", str(nf), "--in_nodefile_type", "SN",
           "--in_fmt", IN_FMT,
           "--in_data_list", str(in_list), "--out_data_list", str(out_list),
           "--var", var, "--bydist", str(cap), "--regional",
           "--latname", LATNAME, "--lonname", LONNAME,
           "--logdir", str(nd)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode !=0:
        return f"failed:rc{r.returncode}", 0
    made = sum(p.exists() for p in outs)
    if made != len(outs):
        return f"failed:wrote{made}of{len(outs)}", made
    return "ok", made

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def attribute(product="era5", cap=CAP_RADIUS, out_root = OUT_ROOT,
              season_min=None, season_max=None, dry_run=False, overwrite=False,
              interpolate=INTERPOLATE_MISSING_00Z):
    _,_, var = PRODUCTS[product]
    work = Path(out_root) / f"{product}_cap{cap:g}"
    work.mkdir(parents=True, exist_ok=True)

    man, stat = build_manifest(product, interpolate=interpolate)
    if season_min or season_max:
        s = np.where(man.spc_day.dt.month == 12,
                     man.spc_day.dt.year + 1, man.spc_day.dt.year)
        m = np.ones(len(man), bool)
        if season_min: m &= s >= season_min
        if season_max: m &= s <= season_max
        man = man[m]

    n_etc = man.track_id.nunique()
    print(f"product={product}  var={var}  cap={cap} deg  ->  {work}")
    print(f"  ETCs to process : {n_etc}")
    print(f"  ETC-day masks   : {len(man):,}  "
          f"({int(man.interpolated.sum())} from interpolated 00Z positions)")
    print("  excluded ETCs   : " +
          ", ".join(f"{k}={v}" for k, v in
                    stat[stat.status != "ok"].status.value_counts().items()))
    if dry_run:
        print(f"\nDry run -- nothing executed")
        return man, stat

    results = []
    for i, (tid, g) in enumerate(man.groupby("track_id"), 1):
        s, n = run_etc(int(tid), g, var, cap, work, overwrite)
        results.append((int(tid), s, n))
        if i % 250 == 0:
            print(f" ... {i}/{n_etc}")

    res = pd.DataFrame(results, columns=["track_id", "run_status", "n_masks"])
    stat = stat.merge(res, on="track_id", how="left")
    stat["run_status"] = stat.run_status.fillna("not_attributed")
    stat["n_masks"] = stat.n_masks.fillna(0).astype(int)
    stat.to_csv(work / "attribution_manifest.csv", index=False)
    man.assign(in_path=man.in_path.astype(str)).to_csv(work / "etc_day_index.csv",
                                                       index=False)

    print(f"\n======== summary ==========")
    for k, v in res.run_status.value_counts().items():
        print(f"  {k:22s}: {v}")
    print(f"  masks written         : {int(res.n_masks.sum()):,}")
    print(f"  manifest -> {work / 'attribution_manifest.csv'}")
    return man, stat

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=sorted(PRODUCTS), default="era5")
    p.add_argument("--cap", type=float, default=CAP_RADIUS)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--season-min", type=int)
    p.add_argument("--season-max", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-interpolate", action="store_true",
                   help="use only 00Z nodes TE emitted; do not fill track gaps")
    a = p.parse_args()

    attribute(a.product, a.cap, a.out_root, a.season_min, a.season_max,
              a.dry_run, a.overwrite, interpolate=not a.no_interpolate)
