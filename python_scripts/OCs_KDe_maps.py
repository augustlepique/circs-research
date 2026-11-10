"""
Quick-look maps of every outbreak cyclone (OC): track, storm reports, KDE cores
and storm-total snowfall, one PNG per cyclone.

Plot style follows jupyter_notebooks/ETC_reports_snowfall_plot.ipynb
(plot_track_case) with the per-core KDE contours from
jupyter_notebooks/ETCs_KDE.ipynb (plot_track_cores):
  * snowfall contourf (Tomer Burg levels/colours)
  * track line + red 'L' every 6 h
  * reports in KEPT cores coloured by type, all other reports in grey
  * one KDE contour (pdf_thresh) per kept core, labelled with its core_id

Files are written as  <outdir>/track<ID>_<genesis YYYYMMDD>_<group>.png

Usage:
    python OCs_KDe_maps.py                       # all OCs, auto snow product
    python OCs_KDe_maps.py --snow era5 --workers 8
    python OCs_KDe_maps.py --tracks 6063 25      # just a few
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as col
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KernelDensity

DATA = Path('/home1/lepique/circs_research/data')
ETC_SUMMARY = DATA / 'etc_summary_final.csv'
TRACKS = DATA / 'nodes_with_reports_final.csv'
REPORTS = DATA / 'reports_tagged_final.csv'
CORES = DATA / 'kde_core_summary.csv'

SNOW_FILES = {
    "era5":   ("/data1/lepique/etc_snowfall/storm_totals_era5_cap12.nc",
               "storm_total_sf"),
    "nohrsc": ("/data1/lepique/etc_snowfall/storm_totals_nohrsc_cap12.nc",
               "storm_total_snowfall_depth"),
}
CAP_RADIUS = 12.0        # great-circle degrees (Hawcroft et al. 2012)
M_TO_IN = 39.37
SWE_RATIO = 10.0

EXTENT = [-122.0, -72.5, 21.5, 50.0]
CLEVS = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48, 60]
COLORS = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c', '#082694',
          '#ffff96', '#ffc400', '#ff8700', '#db1400', '#9e0000',
          '#690000', '#ccccff', '#9f8cd8', '#7c52a5', '#561c72']

# KDE settings -- must match the ones used to build kde_core_summary.csv
BANDWIDTH = 1.0
PDF_THRESH = 0.001
DX = 0.25
PAD = 2.0


## ---------------------------------------------------------------------------
## data
## ---------------------------------------------------------------------------
def load_data():
    etc = pd.read_csv(ETC_SUMMARY, parse_dates=['track_start', 'track_end'])
    ocs = etc[etc['scs_outbreak'] == True].copy()
    ocs['track_id'] = ocs['track_id'].astype(int)

    tracks = pd.read_csv(TRACKS, parse_dates=['time_hour'])
    tracks = tracks[tracks['track_id'].isin(ocs['track_id'])]

    reports = pd.read_csv(REPORTS, low_memory=False)
    reports = reports.dropna(subset=['track_id']).copy()
    reports['track_id'] = reports['track_id'].astype(int)
    reports = reports[reports['track_id'].isin(ocs['track_id'])]
    reports['time_hour'] = pd.to_datetime(reports['time_hour'], utc=True)

    cores = pd.read_csv(CORES)
    cores = cores[cores['kept'] == True]
    return ocs, tracks, reports, cores


def load_snow_fields(track_ids, product):
    """
    Storm-total snowfall (inches, -180..180 lon) for each track, read once up front.

    product = 'nohrsc' | 'era5' | 'auto' (NOHRSC where available, i.e. 2008+,
    otherwise ERA5 at 10:1) | 'none'.
    Returns {track_id: (DataArray or None, label, meta)}.
    """
    out = {int(t): (None, None, None) for t in track_ids}
    if product == 'none':
        return out
    order = ['nohrsc', 'era5'] if product == 'auto' else [product]

    for prod in order:
        path, var = SNOW_FILES[prod]
        with xr.open_dataset(path) as ds:
            todo = [t for t in out if out[t][0] is None and t in ds.track_id]
            if not todo:
                continue
            sub = ds.sel(track_id=todo).load()
        for t in todo:
            s = sub.sel(track_id=t)
            meta = {"product": prod,
                    "status": str(s.status.values),
                    "n_days": int(s.n_days),
                    "first": str(s.first_spc_day.values),
                    "last": str(s.last_spc_day.values)}
            if meta["status"] != "ok" or meta["n_days"] == 0:
                continue
            da = s[var]
            da = da.assign_coords(longitude=da.longitude - 360.0) * M_TO_IN
            if prod == 'era5':
                da = da * SWE_RATIO
                label = f"ERA5 storm-total snowfall (in, {SWE_RATIO:g}:1 from SWE)"
            else:
                label = "NOHRSC storm-total snowfall (in, depth)"
            out[t] = (da.where(da >= CLEVS[0]), label, meta)
    return out


## ---------------------------------------------------------------------------
## plotting helpers
## ---------------------------------------------------------------------------
def in_view(ax, lon, lat):
    """Points inside the axes' current extent, tested in projected coordinates."""
    xy = ax.projection.transform_points(ccrs.PlateCarree(),
                                        np.asarray(lon, dtype=float),
                                        np.asarray(lat, dtype=float))
    x, y = xy[:, 0], xy[:, 1]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (np.isfinite(x) & np.isfinite(y)
            & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))


def core_density(xs, ys):
    """KDE of one core's reports on its padded bounding-box grid (border zeroed)."""
    gx = np.arange(xs.min() - PAD, xs.max() + PAD + DX, DX)
    gy = np.arange(ys.min() - PAD, ys.max() + PAD + DX, DX)
    GX, GY = np.meshgrid(gx, gy)
    kde = KernelDensity(bandwidth=BANDWIDTH, kernel='gaussian')
    kde.fit(np.column_stack([xs, ys]))
    dens = np.exp(kde.score_samples(np.column_stack([GX.ravel(), GY.ravel()])))
    dens = dens.reshape(GX.shape)
    dens[[0, -1], :] = 0
    dens[:, [0, -1]] = 0
    return GX, GY, dens


def plot_oc(track_id, oc_row, case_track, case_reports, case_cores,
            snow, snow_label, snow_meta, savepath, dpi):
    transform = ccrs.PlateCarree()
    proj_lcc = ccrs.LambertConformal(central_longitude=-99.0, central_latitude=35.0,
                                     standard_parallels=[35.0])
    fig = plt.figure(figsize=(12, 8.5))
    ax = plt.axes(projection=proj_lcc)

    ## geography
    ax.coastlines(resolution='50m', color='black', linewidths=0.8)
    ax.add_feature(cfeature.BORDERS, linewidths=0.8, edgecolor='black')
    state_borders = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lakes',
        scale='50m', facecolor='none')
    ax.add_feature(state_borders, edgecolor='black', linewidth=0.5, zorder=4)
    ax.set_extent(EXTENT, crs=transform)

    ## --- 1. snowfall ---
    if snow is not None:
        cmap = col.ListedColormap(COLORS[:len(CLEVS) - 1])
        cmap.set_over(COLORS[-1])
        norm = col.BoundaryNorm(CLEVS, cmap.N)
        cs = ax.contourf(snow.longitude.values, snow.latitude.values, snow.values,
                         levels=CLEVS, cmap=cmap, norm=norm, extend='max',
                         transform=transform, zorder=1)
        cbar = plt.colorbar(cs, ax=ax, shrink=0.6, pad=0.02, ticks=CLEVS)
        cbar.set_label(snow_label, fontsize=11)

    ## --- 2. track line + 'L' every 6 h ---
    ax.plot(case_track['lon'], case_track['lat'], color='black', linewidth=2,
            alpha=0.8, transform=transform, zorder=5)
    track_6h = case_track[case_track['time_hour'].dt.hour % 6 == 0]
    visible = in_view(ax, track_6h['lon'], track_6h['lat'])
    for _, row in track_6h[visible].iterrows():
        ax.text(row['lon'], row['lat'], 'L', fontsize=13, fontweight='bold',
                color='red', ha='center', va='center', transform=transform,
                zorder=5, clip_on=True)

    ## --- 3. KDE contour per kept core ---
    keep = case_reports['core_kept'] == True
    palette = plt.cm.tab10(np.arange(len(case_cores)) % 10)
    for color, (_, c) in zip(palette, case_cores.iterrows()):
        gc = case_reports[case_reports['core_id'] == c['core_id']]
        xs, ys = gc['report_lon'].to_numpy(), gc['report_lat'].to_numpy()
        if len(xs) < 2:
            continue
        GX, GY, dens = core_density(xs, ys)
        ax.contour(GX, GY, dens, levels=[PDF_THRESH], colors=[color],
                   linewidths=1.8, transform=transform, zorder=6)
        t0 = pd.Timestamp(c['t_start'])
        ax.plot([], [], color=color, linewidth=1.8,
                label=f"{c['core_id']}: {int(c['n_reports'])} rpts, "
                      f"{t0:%m-%d %H}Z +{c['duration_h']:.0f}h")

    ## --- 4. storm reports ---
    et = case_reports['EVENT_TYPE'].str.upper().fillna('OTHER')
    is_tor = et.str.contains('TORNADO')
    is_hail = et.str.contains('HAIL')
    is_wind = et.str.contains('THUNDERSTORM WIND')
    ax.scatter(case_reports.loc[~keep, 'report_lon'], case_reports.loc[~keep, 'report_lat'],
               s=12, c='grey', alpha=0.5, edgecolor='black', linewidth=0.2,
               transform=transform, zorder=6, label='Pruned / no core')
    ax.scatter(case_reports.loc[is_wind & keep, 'report_lon'],
               case_reports.loc[is_wind & keep, 'report_lat'],
               s=15, c='blue', alpha=0.7, edgecolor='black', linewidth=0.2,
               transform=transform, zorder=7, label='T-Storm Wind')
    ax.scatter(case_reports.loc[is_hail & keep, 'report_lon'],
               case_reports.loc[is_hail & keep, 'report_lat'],
               s=15, c='lime', alpha=0.7, edgecolor='black', linewidth=0.2,
               transform=transform, zorder=8, label='Hail')
    ax.scatter(case_reports.loc[is_tor & keep, 'report_lon'],
               case_reports.loc[is_tor & keep, 'report_lat'],
               s=40, c='red', alpha=0.7, marker='v', edgecolor='black', linewidth=0.5,
               transform=transform, zorder=9, label='Tornado')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    ## --- title ---
    t0 = case_track['time_hour'].min().strftime('%Y-%m-%d %H:%M')
    t1 = case_track['time_hour'].max().strftime('%Y-%m-%d %H:%M')
    title = (f"Track {track_id} ({oc_row['composite_group']}) | {t0} → {t1} UTC | "
             f"min MSL {case_track['msl'].min():.1f} hPa\n"
             f"{len(case_reports)} reports, {int(keep.sum())} in "
             f"{len(case_cores)} kept core(s)")
    if snow_meta:
        title += (f" | snow: {snow_meta['product'].upper()} {snow_meta['first']} → "
                  f"{snow_meta['last']} ({snow_meta['n_days']} SPC days)")
    else:
        title += " | no snowfall attributed"
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

    fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


## ---------------------------------------------------------------------------
## driver (globals filled in main() so forked workers inherit them)
## ---------------------------------------------------------------------------
_CTX = {}


def _plot_one(track_id):
    c = _CTX
    oc_row = c['ocs'].loc[track_id]
    case_track = c['tracks'].get_group(track_id).sort_values('time_hour')
    case_reports = c['reports'].get_group(track_id)
    case_cores = (c['cores'].get_group(track_id).sort_values('t_start')
                  if track_id in c['cores'].groups else c['cores_empty'])
    snow, snow_label, snow_meta = c['snow'][track_id]

    genesis = pd.Timestamp(oc_row['track_start'])
    fname = (f"track{track_id:05d}_{genesis:%Y%m%d}_{oc_row['composite_group']}.png")
    savepath = c['outdir'] / fname
    if savepath.exists() and not c['overwrite']:
        return f"skip {fname}"
    try:
        plot_oc(track_id, oc_row, case_track, case_reports, case_cores,
                snow, snow_label, snow_meta, savepath, c['dpi'])
    except Exception as err:          # one bad case shouldn't kill the batch
        plt.close('all')
        return f"FAIL track {track_id}: {err!r}"
    return f"wrote {fname}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', default='/home1/lepique/circs_research/Figs/OC_kde_maps')
    p.add_argument('--snow', default='auto', choices=['auto', 'nohrsc', 'era5', 'none'],
                   help="snowfall product (auto = NOHRSC where available, else ERA5)")
    p.add_argument('--dpi', type=int, default=80)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tracks', type=int, nargs='*', help='only these track_ids')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ocs, tracks, reports, cores = load_data()
    if args.tracks:
        ocs = ocs[ocs['track_id'].isin(args.tracks)]
    track_ids = ocs['track_id'].tolist()
    print(f"{len(track_ids)} outbreak cyclones -> {outdir}")

    _CTX.update(ocs=ocs.set_index('track_id'),
                tracks=tracks.groupby('track_id'),
                reports=reports.groupby('track_id'),
                cores=cores.groupby('track_id'),
                cores_empty=cores.iloc[0:0],
                snow=load_snow_fields(track_ids, args.snow),
                outdir=outdir, dpi=args.dpi, overwrite=args.overwrite)

    if args.workers > 1:
        # explicit fork: Python 3.14 defaults to forkserver, which would not
        # inherit _CTX
        with mp.get_context("fork").Pool(args.workers) as pool:
            for msg in pool.imap_unordered(_plot_one, track_ids):
                print(msg, flush=True)
    else:
        for tid in track_ids:
            print(_plot_one(tid), flush=True)


if __name__ == '__main__':
    main()
