"""
snowfall_index.py
=================
A population - free, CONUS-uniform snowfall severity index for each tracked ETC,
adapted from the Regional Snowfall Index (Squires et al. 2014, BAMS)

    Index_k = SUM_i A_{k, i} / Abar_i
    
where A_{k, i} is the true (lat weighted) area within storm k's Hawcroft cap (12 degrees)
footprint whose storm-total snow depth meets or sxceeds tier T_i, and Abar_i 
is the mean of A_{k, i} across the climatology. RSI's population term is dropped,
so this only measures extent x amount and not realized desruption (deliberately)

Depth comes from the ERA5 storm total SWE field via constant snow to liquid
ratio of 10. This is done for simplicity, and a constant ratio is linear and commutes
with the lifecycle sum, and is applied directly to the stage 3 total. However,
this does undercount dry cold western/rockies depth, but it is acceptable for a coarse
significant/not significant split (but not for quantitative use!) 

CONUS MASKING
-------------
Most of storm's total depth-volume lies outside CONUS (for ERA5). IF unmasked,
index is dominated by snowfall over the ocean or Canada. Mask is derived from the 
NOHRSC analysis footpprint. 

Properties worth knowing before index interpretation:
----------------------------------------------------
- Duration leaks into extent. A storm total is the sum of daily caps over the 
lifecycle, so a slow 10-day storm sweeps a far larger footprint than a quick
two day storm of equal intensity. True for regular RSI index as well.
- Climatology is self referential and built from overlapping footprints. 
Caps of co0existing cyclones overlap so the same snowfall can enter several storms'
areas. A(bar) is thus inflated. 

Usage
------
    python snowfall_index.py --sweep                # tier sensitivity, writes nothing
    python snowfall_index.py --dry-run               # full calc, prints, writes nothing
    python snowfall_index.py --tiers 3 6 12 18       # compute + attach to etc_summary
    python snowfall_index.py --product nohrsc --dry-run     # validates 10:1 SLR if desired
"""

import argparse
import glob
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ETC_SUMMARY = REPO_ROOT / "data" / "etc_summary_final.csv"
SNOW_ROOT = Path("/data1/lepique/etc_snowfall")
NOHRSC_DAILY = Path("/data1/lepique/nohrsc_daily/")
MASK_CACHE = SNOW_ROOT / "conus_mask.nc"

CAP = 12.0

# products -> (storm total file stem, variable, is_already_depth)
PRODUCTS = {
    "era5":   ("storm_totals_era5_cap{cap:g}.nc", "storm_total_sf", False),
    "nohrsc": ("storm_totals_nohrsc_cap{cap:g}.nc", "storm_total_snowfall_depth", True),
}

# depth tiers, INCHES. CONUS-uniform (differs from RSI which differs by region)
TIERS_IN = (4.0, 10.0, 20.0, 30.0)  # thresholds for "significant" snowfall

SLR = 10   # snow to liquid ratio
CONUS_MASK = True

# "all"  -> Abar_i = mean over every climatology storm, zeros included 
# "topN" -> Abar_i = mean of the top N nonzero values, as Squires et al. do (N=75)
NORMALIZE = "topN"
TOP_N = 75

# Climatology window, DJFM season labelled by JFM year
SEASON_MIN, SEASON_MAX = 1996, 2025

# significant_snow rule: ("index", value) or ("percentile", 0.0-1.0 of non NaNs).
# Index >= 1.0 means this storm covered at least a climatological storm's worth
# of area above T1 (natural analogue of RSI's category 1 boundary)
SIGNIFICANCE = ("index", 0.5)

OUTBREAK_MIN_CORES = 1

M_PER_IN = 0.0254
R_EARTH_KM = 6371.0088  # radius of earth in km

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def cell_area_km2(lat, lon):
    """
    True spherical cell area, km^2, broadcast to (nlat, nlon)
    Area = R^2 * dlon * [sin(phi + dphi/2) - sin(phi-dphi/2)]
    """
    dlat = abs(np.diff(lat).mean())
    dlon = abs(np.diff(lon).mean())

    band = (R_EARTH_KM**2 * np.radians(dlon)
            * np.abs(np.sin(np.radians(lat + dlat/2))
                     - np.sin(np.radians(lat - dlat/2))))
    return band[:, None] * np.ones((1, len(lon)))

def conus_mask(lat, lon, cache=MASK_CACHE, nohrsc_dir=NOHRSC_DAILY, sample=200):
    """
    boolean CONUS mask on the storm total grid, from the NOHRSC analysis footprint.
    
    NOHRSC is NaN outside its analysis domain, and after stage 1b regridding it
    sits exactly on this grid, so the union of its valid cells is already a CONUS
    mask. Cached to disk b/c it is always the same.
    """
    if Path(cache).exists():
        with xr.open_dataset(cache) as d:
            m = d.conus.values.astype(bool)
        if m.shape == (len(lat), len(lon)):
            return m
        print (f" [warn] cached mask shape {m.shape} !=grid; rebuilding")

    files = sorted(Path(nohrsc_dir).glob("daily_snow_nohrsc_*.nc"))
    if not files:
        sys.exit(f"ERROR: no NOHRSC files found in {nohrsc_dir} to build the CONUS mask")

    m = None
    for f in files[::sample]:
        with xr.open_dataset(f) as d:
            v = np.isfinite(d.snowfall_depth.isel(time=0).values)
        m = v if m is None else (m | v)
    xr.Dataset({"conus": (("latitude", "longitude"), m)},
               coords={"latitude": lat, "longitude": lon}).to_netcdf(cache)
    print(f" build CONUS mask ({int(m.sum())} cells) -> {cache}")
    return m 

# ---------------------------------------------------------------------------
# Pass 1 -- areas above each tier
# ---------------------------------------------------------------------------
def tier_areas(product="era5", cap=CAP, tiers_in=TIERS_IN, slr=SLR,
               use_mask=CONUS_MASK, snow_root=SNOW_ROOT):
    """
    a[k, i] in km^2, plus the per storm attribution state.
    
    Returns (track_ids, A, attributed, n_days). 'attributed' is FAlse for storms
    stage 3 stored as all-NaN, which must be held apart from storms that were
    measured and simply never reached T1.
    """
    stem, var, already_depth = PRODUCTS[product]
    path = Path(snow_root) / stem.format(cap=cap)
    if not path.exists():
        sys.exit(f"ERROR: missing {path} (probably need to run accumulate_storm_totals.py)")

    ds = xr.open_dataset(path)
    lat, lon = ds.latitude.values, ds.longitude.values
    area = cell_area_km2(lat, lon)
    mask = conus_mask(lat, lon) if use_mask else np.ones(area.shape, bool)
    aw = area * mask                ## zero weights outside of conus

    tids = ds.track_id.values.astype(int)
    attributed = ds.n_days.values > 0
    n_days = ds.n_days.values
    tiers_m = np.asarray(tiers_in, float) * M_PER_IN

    A = np.zeros((len(tids), len(tiers_m)), float)
    vals = ds[var].values             # (track, lat, lon)
    for k in range(len(tids)):
        if not attributed[k]:
            A[k] = np.nan
            continue 
        depth = vals[k] if already_depth else vals[k] * slr
        for i, T in enumerate(tiers_m):
            A[k, i] = float(aw[depth >=T].sum())
    ds.close()

    return tids, A, attributed, n_days

# -----------------------------------------------------------------------------
# Pass 2: normalize and index
# -----------------------------------------------------------------------------
def season_year(ts):
    ts = pd.Timestamp(ts)
    return ts.year + 1 if ts.month == 12 else ts.year

def normalize(A, attributed, in_climo, how=NORMALIZE, top_n=TOP_N):
    """
    Abar_i and the normalized areas a[k, i] = A[k, i] / Abar_i

    how="all" averages over every climatology storm INCLUDING zeros. 
    This has an effect though, wherea. tier met by only ~4% of storms gets a 
    very small Abar, so  the few storms that do meet it get very large 
    a_{k,4}, and the index is driven hard by the top tier, which is the RSI's
    original intent, but is more extreme, as RSI averages the top 75 NONZERO
    values precisely to keep the denominators stable. how="topN" reproduces that. 
    """
    use = attributed & in_climo
    if use.sum() == 0:
        sys.exit("ERROR: no storms in the climatology window)")
    Abar = np.empty(A.shape[1])
    for i in range(A.shape[1]):
        col = A[use, i]
        if how == "all":
            Abar[i] = col.mean()
        elif how == "topN":
            nz = np.sort(col[col > 0])[::-1][:top_n]
            Abar[i] = nz.mean() if len(nz) else np.nan
        else:
            sys.exit(f"ERROR: unknown NORMALIZE={how!r}")
    if not np.all(np.isfinite(Abar)) or np.any(Abar <= 0):
        bad = [i for i, v in enumerate(Abar) if not np.isfinite(v) or v <= 0]
        sys.exit(f"ERROR: tier(s) {bad} are never exceeded in the climatology;"
                 f"lower the tiers, or widen the window")

    a = A / Abar

    return Abar, a

def build_index(a,attributed):
    """
    Index_k = sum_i a[k, i] with the two NaN cases kept apart.
    
    reason "not_attributed" -> no storm total field at all
    reason "no_tier_exceeded" -> measured but never reached T1 anywhere in CONUS
    
    Both leave Index NaN, but they are different statements (missing data, and that 
    storm was small/not significant). Only the second can be resolved to significant_snow = False)
    """
    idx = np.nansum(a, axis=1)
    reason = np.array(["ok"] * len(idx), dtype=object)
    idx[~attributed] = np.nan
    reason[~attributed] = "not_attributed"
    none_met = attributed & (np.nan_to_num(a).sum(axis=1) == 0)
    idx[none_met] = np.nan
    reason[none_met] = "no_tier_exceeded"
    return idx, reason 


def significance(idx, reason, rule=SIGNIFICANCE):
    """
    Boolean significant_snow, with pd.NA where it is genuinely undetermined
    
    no_tier_exceeded -> False (measured, too small)
    not_attributed -> pd.NA (we don't know)
    """
    kind, val = rule
    ok = np.isfinite(idx)
    if kind == "index":
        thr = float(val)
    elif kind == "percentile":
        thr = float(np.quantile(idx[ok], val)) if ok.any() else np.nan
    else:
        sys.exit(f"ERROR: unknown significance rule {kind!r}")
    flag = pd.array([pd.NA] * len(idx), dtype="boolean")
    flag[ok] = idx[ok] >= thr
    flag[reason == "no_tier_exceeded"] = False
    return flag, thr

def composite_group(sig, outbreak):
    """Four way crossing of the SCS outbreak flag with significant_snow"""
    g = pd.Series(pd.NA, index=range(len(sig)), dtype="object")
    known = ~pd.isna(sig)
    s = sig.to_numpy(dtype="object")
    for i in np.where(known)[0]:
        g[i] = (("OC_SNOW" if s[i] else "OC_ONLY") if outbreak[i]
                else ("SNOW_ONLY" if s[i] else "NEITHER"))
    return g


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def compute(product="era5", cap=CAP, tiers_in=TIERS_IN, slr=SLR,
            use_mask=CONUS_MASK, how=NORMALIZE, rule=SIGNIFICANCE,
            etc_summary=ETC_SUMMARY):

    summ = pd.read_csv(etc_summary)
    summ['track_id'] = summ.track_id.astype(int)

    tids, A, attributed, n_days = tier_areas(product, cap, tiers_in, slr, use_mask)
    sy = pd.to_datetime(summ.time_min_msl, utc=True).apply(season_year)
    sy_by_tid = dict(zip(summ.track_id, sy))
    in_climo = np.array([SEASON_MIN <= sy_by_tid.get(t, -1) <= SEASON_MAX
                        for t in tids])

    Abar, a = normalize(A, attributed, in_climo, how)
    idx, reason = build_index(a, attributed)
    sig, thr = significance(idx, reason, rule)

    out = pd.DataFrame({"track_id": tids,
                        "snow_index": idx,
                        "snow_index_reason": reason,
                        "significant_snow": sig,
                        "snow_n_days": n_days})
    for i, T in enumerate(tiers_in):
        out[f"snow_area_T{i+1}_km2"] = np.where(attributed, A[:,i], np.nan)
        out[f"snow_a_T{i+1}"] = np.where(attributed, a[:,i], np.nan)

    merged = summ.merge(out, on="track_id", how='left')
    outbreak = (merged.n_cores_kept.fillna(0) >= OUTBREAK_MIN_CORES).to_numpy()
    merged["scs_outbreak"] = outbreak
    merged["composite_group"] = composite_group(merged.significant_snow, outbreak).values

    return merged, Abar, thr, tiers_in

def report(merged, Abar, thr, tiers_in, product, how):
    ok = merged.snow_index.notna()
    print(f"product={product}  tiers={tuple(tiers_in)} in  normalize={how}")
    print(f"  Abar (km2) : " + "  ".join(f"T{i+1}={v:,.0f}" for i, v in enumerate(Abar)))
    print(f"  storms     : {len(merged)}  index computed {int(ok.sum())}")
    print("  NaN reasons: " + ", ".join(
        f"{k}={v}" for k, v in merged.loc[~ok, 'snow_index_reason'].value_counts().items()))
    if ok.any():
        q = merged.loc[ok, "snow_index"].quantile([.5, .75, .9, .99]).round(2).to_dict()
        print(f"  index      : median {q[0.5]}  p75 {q[0.75]}  p90 {q[0.9]}  "
              f"p99 {q[0.99]}  max {merged.snow_index.max():.1f}")
    print(f"  significance threshold: index >= {thr:.3f}")
    print(f"  significant_snow: True={int((merged.significant_snow==True).sum())}  "
        f"False={int((merged.significant_snow==False).sum())}  "
        f"NA={int(merged.significant_snow.isna().sum())}")
    print("\n  composite groups:")
    for k, v in merged.composite_group.value_counts(dropna=False).items():
        print(f"  {str(k):10s}: {v}")


def sweep(product="era5", cap=CAP, slr=SLR, use_mask=CONUS_MASK, how=NORMALIZE,
          rule=SIGNIFICANCE, tier_sets=((2, 5, 10, 15), (3, 6, 12, 18),
                                        (4, 10, 20, 30), (3, 7, 14, 21))):
    """Tier sensitivity: how each candidate set splits the population."""
    print("tier sensitivity (nothing written)\n")
    for ts in tier_sets:
        m, Abar, thr, _ = compute(product, cap, ts, slr, use_mask, how, rule)
        vc = m.composite_group.value_counts()
        ok = m.snow_index.notna()
        print(f"  {str(tuple(ts)):>18s} in -> sig={int((m.significant_snow==True).sum()):4d}  "
              f"median_index={m.loc[ok,'snow_index'].median():5.2f}  "
              + "  ".join(f"{g}={vc.get(g,0)}" for g in
                          ("OC_SNOW", "OC_ONLY", "SNOW_ONLY", "NEITHER")))


## main driver
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=sorted(PRODUCTS), default="era5")
    p.add_argument("--cap", type=float, default=CAP)
    p.add_argument("--tiers", type=float, nargs=4, default=list(TIERS_IN),
                   metavar=("T1", "T2", "T3", "T4"), help="depth tiers, inches")
    p.add_argument("--slr", type=float, default=SLR)
    p.add_argument("--no-mask", action="store_true", help="do NOT restrict to CONUS")
    p.add_argument("--normalize", choices=("all", "topN"), default=NORMALIZE)
    p.add_argument("--sig-index", type=float, help="significant_snow if index >= this")
    p.add_argument("--sig-percentile", type=float,
                   help="significant_snow if index >= this quantile of non-NaN")
    p.add_argument("--etc-summary", type=Path, default=ETC_SUMMARY)
    p.add_argument("--out", type=Path, help="write here instead of in place")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sweep", action="store_true")
    a = p.parse_args()

    rule = (("percentile", a.sig_percentile) if a.sig_percentile is not None
            else ("index", a.sig_index) if a.sig_index is not None
            else SIGNIFICANCE)

    if a.sweep:
        sweep(a.product, a.cap, a.slr, not a.no_mask, a.normalize, rule)
        sys.exit(0)

    merged, Abar, thr, tiers = compute(a.product, a.cap, tuple(a.tiers), a.slr,
                                       not a.no_mask, a.normalize, rule,
                                       a.etc_summary)
    report(merged, Abar, thr, tiers, a.product, a.normalize)

    if a.dry_run:
        print("\nDry-run -- nothing written.")
        sys.exit(0)

    dest = a.out or a.etc_summary
    if dest == a.etc_summary:
        # overwriting a file that has accumulated columns across many passes
        # (ONI, MJO, RI_*): snapshot it first so a bad run is recoverable
        bak = a.etc_summary.with_suffix(
            f".{datetime.now():%Y%m%dT%H%M%S}.bak.csv")
        shutil.copy2(a.etc_summary, bak)
        print(f"\n backup -> {bak}")
    merged.to_csv(dest, index=False)
    print(f" wrote -> {dest} ({len(merged.columns)} columns)")




