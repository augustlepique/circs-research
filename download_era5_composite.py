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

From a notebook (not recommended though):
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
import threading
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
COL_TIMEMAXREP = "time_max_reports"
COL_NCORESKEPT = "n_cores_kept"

# Outbreak cyclones (>= this many kept cores) additionally get their single
# hour of maximum reports added to the download (one time step, no offsets).
OUTBREAK_MIN_CORES = 1

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
def add_max_report_time(etc_summary, node_reports,
                        track_col=COL_TRACKID, time_col="time_hour",
                        rep_col="n_reports", out_col=COL_TIMEMAXREP):
    """
    Add a ``time_max_reports`` column to ``etc_summary``: for each track, the
    hour of its maximum report count (ties -> earliest). Tracks with zero
    reports get NaT. Mirrors the computation in the compositing notebook, and
    is only needed for the CLI path (the notebook's in-memory etc_summary
    already carries this column).
    """
    nr = node_reports.copy()
    nr[time_col] = pd.to_datetime(nr[time_col], utc=True)
    top = (nr.sort_values([track_col, rep_col, time_col],
                          ascending=[True, False, True])
             .groupby(track_col, as_index=False).head(1))
    top = top[top[rep_col] > 0]
    tmax = top.set_index(track_col)[time_col]
    es = etc_summary.copy()
    es[out_col] = es[track_col].map(tmax)
    return es


def season_year(ts):
    """DJFM season label: December belongs to the following year's JFM season."""
    ts = pd.Timestamp(ts)
    return ts.year + 1 if ts.month == 12 else ts.year


def filter_by_season(etc_summary, season_min=None, season_max=None,
                     time_col=COL_TIMEMSL):
    """
    Keep only cyclones whose reference-time DJFM season is within
    ``[season_min, season_max]`` (inclusive); either bound may be None (open).
    Season year = calendar year of JFM, with December attributed to the
    following year's season. Lets the CLI download one season at a time.
    """
    if season_min is None and season_max is None:
        return etc_summary
    sy = pd.to_datetime(etc_summary[time_col], utc=True).apply(season_year)
    mask = pd.Series(True, index=etc_summary.index)
    if season_min is not None:
        mask &= sy >= season_min
    if season_max is not None:
        mask &= sy <= season_max
    return etc_summary[mask]


def timestamps_from_etc_summary(
    etc_summary,
    time_col=COL_TIMEMSL,
    track_col=COL_TRACKID,
    offsets=OFFSET_HOURS,
    season_months=SEASON_MONTHS,
    include_max_reports=True,
    maxrep_col=COL_TIMEMAXREP,
    cores_col=COL_NCORESKEPT,
    outbreak_min_cores=OUTBREAK_MIN_CORES,
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
            records.append({"track_id": tid, "ref": "min_msl",
                            "offset_h": int(off), "timestamp": ts})

    # Outbreak cyclones (>= outbreak_min_cores kept cores) additionally get the
    # single hour of maximum reports -- one time step, no offsets. Skipped
    # silently if the required columns are absent (e.g. a bare etc_summary).
    if include_max_reports and maxrep_col in es.columns and cores_col in es.columns:
        oc = es[es[cores_col] >= outbreak_min_cores]
        for tid, ts in zip(oc[track_col], pd.to_datetime(oc[maxrep_col], utc=True)):
            if pd.notna(ts):
                records.append({"track_id": tid, "ref": "max_reports",
                                "offset_h": 0, "timestamp": ts})

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
    if "ref" in expanded.columns:
        n_mr = int((expanded["ref"] == "max_reports").sum())
        print(f"  max-reports refs      : {n_mr}  (outbreak cyclones, +0h)")

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
_thread_local = threading.local()


def _get_client():
    """
    Return this thread's cdsapi client, creating it once and reusing it.

    Reuse is essential: constructing a new ``cdsapi.Client()`` per request opens
    a fresh HTTPS session each time and leaks its sockets, exhausting the process
    open-file limit ("Too many open files") after ~1000 requests. With a
    thread-local client there are only ~``workers`` sessions total, each pooling
    and reusing its connections, so the descriptor count stays bounded.
    """
    client = getattr(_thread_local, "client", None)
    if client is None:
        import cdsapi  # lazy: keep the module importable without cdsapi installed
        client = cdsapi.Client()
        _thread_local.client = client
    return client


def _retrieve_one(dataset, request, target):
    """
    Submit a single CDS request and save to ``target``.

    Downloads to a ``.part`` sidecar first and renames on success, so an
    interrupted/failed request can never leave a partial file that the
    skip-existing logic would later mistake for a completed download. Reuses
    this thread's cdsapi client (see ``_get_client``) to avoid leaking sockets.
    """
    tmp = target.with_name(target.name + ".part")
    _get_client().retrieve(dataset, request, str(tmp))
    tmp.replace(target)  # same-dir rename: only a complete file gets the real name
    return target.name


def _download_via_cdsswarm(tasks, workers):
    """
    Submit ``tasks`` (list of (dataset, request, target) tuples) through the
    ``cdsswarm`` package instead of our own thread pool. EXPERIMENTAL -- only
    invoke this on a limited trial (e.g. one season) before trusting it for a
    full run; see caveats below.

    cdsswarm's own worker loop (cdsswarm.core._download_one) constructs a new
    ``cdsapi.Client()`` per task rather than reusing one per worker thread --
    the same pattern that caused our earlier "Too many open files" leak in
    this script. It has not been confirmed leak-free at the request volumes
    this script needs (thousands), so this path is opt-in and should be
    monitored (e.g. ``ls /proc/<pid>/fd | wc -l``) while trying it.

    Also, cdsapi's underlying ``Result._download`` writes directly to the
    target path rather than a temp file, so an interrupted task could leave a
    partial file at the real filename. To preserve the same atomicity
    guarantee as ``_retrieve_one``, this function downloads to a ``.part``
    sidecar (via the target it hands to cdsswarm) and renames on success.
    """
    import cdsswarm

    part_to_target = {}
    cds_tasks = []
    for dataset, request, target in tasks:
        part = target.with_name(target.name + ".part")
        part.unlink(missing_ok=True)  # any stray .part here is from an incomplete prior attempt
        part_to_target[str(part)] = target
        cds_tasks.append(cdsswarm.Task(dataset=dataset, request=request, target=str(part)))

    results = cdsswarm.download(
        cds_tasks,
        num_workers=workers,
        skip_existing=False,  # we already pre-filtered `tasks` to pending-only
        reuse_jobs=True,      # attach to matching jobs already on the CDS account
        max_retries=3,
        on_message=lambda msg: None,  # silence cdsswarm's own printing; we log below
    )

    for r in results:
        target = part_to_target[r.task.target]
        if r.success:
            Path(r.task.target).replace(target)
            print(f" -> saved {target.name}")
        else:
            print(f" !! FAILED {target.name}: {r.error}")


def download(date_groups, out_dir=OUT_DIR, dry_run=True, manifest=None,
             expanded=None, workers=1, only="both", engine="manual"):
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
    workers     : number of CDS requests to run concurrently via a thread pool.
                  CDS-Beta throttles active requests per account, so keep this
                  small (~3-4); too many just queue or get rejected. workers=1
                  preserves the original strictly-sequential behaviour.
    only        : "both" (default), "sfc", or "plev" -- restrict to the
                  single-level or pressure-level dataset. Lets you download the
                  fast surface files first and the slow pressure-level files in a
                  later pass; the two use different filenames so they never clash.
    engine      : "manual" (default, our own thread pool + persistent client)
                  or "cdsswarm" (delegate to the cdsswarm package for retries
                  and CDS job-reuse). EXPERIMENTAL: see _download_via_cdsswarm
                  docstring for an unconfirmed FD-leak caveat -- try on a
                  small subset first and monitor /proc/<pid>/fd.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defensive: raise the open-file soft limit to absorb transient FD spikes
    # under concurrency (the persistent per-thread client keeps usage low, but
    # this leaves headroom). Harmless if it cannot be raised.
    if not dry_run and workers > 1:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
        except Exception:
            pass

    datasets = [
        ("reanalysis-era5-single-levels", SFC_VARS, None, "era5_sfc"),
        ("reanalysis-era5-pressure-levels", PLEV_VARS, PRESSURE_LEVELS, "era5_plev"),
    ]
    if only == "sfc":
        datasets = [d for d in datasets if d[3] == "era5_sfc"]
    elif only == "plev":
        datasets = [d for d in datasets if d[3] == "era5_plev"]

    # Build the list of pending (dataset, request, target) tasks, skipping any
    # output that already exists.
    tasks = []
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
                tasks.append((dataset, request, target))

    if not dry_run and tasks:
        print(f" submitting {len(tasks)} request(s) with {workers} worker(s) "
              f"[engine={engine}] ...")
        if engine == "cdsswarm":
            _download_via_cdsswarm(tasks, workers)
        elif workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_retrieve_one, ds, rq, tg): tg
                        for ds, rq, tg in tasks}
                for fut in as_completed(futs):
                    tg = futs[fut]
                    try:
                        print(f" -> saved {fut.result()}")
                    except Exception as e:  # keep going; re-run retries this date
                        print(f" !! FAILED {tg.name}: {e}")
        else:
            for ds, rq, tg in tasks:
                print(f" requesting {tg.name} ...")
                try:
                    _retrieve_one(ds, rq, tg)
                    print(f" -> saved {tg.name}")
                except Exception as e:
                    print(f" !! FAILED {tg.name}: {e}")

    # Always refresh bookkeeping so it reflects current/planned out_dir contents.
    write_data_lists(date_groups, out_dir)
    if manifest is not None:
        write_manifest_csv(manifest, out_dir)
    if manifest is not None and expanded is not None:
        print_summary(manifest, expanded, date_groups)


def run(etc_summary, out_dir=OUT_DIR, dry_run=True, workers=1, only="both",
        engine="manual", **kwargs):
    """
    Convenience one-call pipeline: expand timestamps from ``etc_summary``,
    build date groups, and download (or preview). Extra kwargs are forwarded to
    ``timestamps_from_etc_summary`` (e.g. ``time_col``, ``offsets``).

    Returns (manifest, expanded, date_groups).
    """
    manifest, expanded = timestamps_from_etc_summary(etc_summary, **kwargs)
    date_groups = build_date_groups(manifest)
    download(date_groups, out_dir, dry_run=dry_run, manifest=manifest,
             expanded=expanded, workers=workers, only=only, engine=engine)
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
        default="data/etc_summary_final.csv",
        help="CSV with 'track_id' and 'time_min_msl' columns "
             "(default: data/etc_summary_final.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--node-reports",
        default="data/nodes_with_reports_final.csv",
        help="Per-node report-count CSV, used to compute the hour of maximum "
             "reports when 'time_max_reports' is absent from --etc-summary "
             "(default: data/nodes_with_reports_final.csv).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually submit to CDS. Default is a dry run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CDS requests to run concurrently (default: 1). "
             "CDS-Beta limits active requests per account, so keep this small "
             "(3-4 is a safe starting point).",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=None,
        help="Only download cyclones whose DJFM season year is >= this "
             "(December is attributed to the following year's season). "
             "Default: no lower bound.",
    )
    parser.add_argument(
        "--season-max",
        type=int,
        default=None,
        help="Only download cyclones whose DJFM season year is <= this. "
             "For a single season, set --season-min and --season-max equal "
             "(e.g. --season-min 2001 --season-max 2001). Default: no upper bound.",
    )
    parser.add_argument(
        "--only",
        choices=["both", "sfc", "plev"],
        default="both",
        help="Restrict to single-level ('sfc') or pressure-level ('plev') data. "
             "Use 'sfc' for a fast first pass, then 'plev' for the slow second "
             "pass. Default: both.",
    )
    parser.add_argument(
        "--engine",
        choices=["manual", "cdsswarm"],
        default="manual",
        help="Download engine. 'manual' (default) is our own thread pool with "
             "a persistent per-thread cdsapi client. 'cdsswarm' delegates to "
             "the cdsswarm package for retries and CDS job-reuse -- "
             "EXPERIMENTAL, not yet confirmed leak-free at full scale; try on "
             "a small --season-min/--season-max subset first.",
    )
    args = parser.parse_args()

    etc_summary = pd.read_csv(args.etc_summary)
    if COL_TIMEMAXREP not in etc_summary.columns and args.node_reports:
        node_reports = pd.read_csv(args.node_reports)
        etc_summary = add_max_report_time(etc_summary, node_reports)
    etc_summary = filter_by_season(etc_summary, args.season_min, args.season_max)
    if args.season_min is not None or args.season_max is not None:
        lo = args.season_min if args.season_min is not None else "start"
        hi = args.season_max if args.season_max is not None else "end"
        print(f"Season filter {lo}-{hi}: {len(etc_summary)} cyclones selected.")
    run(etc_summary, out_dir=Path(args.out_dir), dry_run=not args.download,
        workers=args.workers, only=args.only, engine=args.engine)
