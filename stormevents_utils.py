#stormevents_utils.py

import pandas as pd
import numpy as np
from pathlib import Path
import re

TIMEZONE_MAP = {
    # Eastern
    "EST":   "America/New_York",
    "EST-5": "America/New_York",
    "EDT":   "America/New_York",

    # Central
    "CST":   "America/Chicago",
    "CST-6": "America/Chicago",
    "CDT":   "America/Chicago",

    # Mountain
    "MST":   "America/Denver",
    "MST-7": "America/Denver",
    "MDT":   "America/Denver",

    # Pacific
    "PST":   "America/Los_Angeles",
    "PST-8": "America/Los_Angeles",
    "PDT":   "America/Los_Angeles",

    # Hawaii (no DST)
    "HST":   "Pacific/Honolulu",
    "HAWAII":"Pacific/Honolulu",

    # Atlantic / Puerto Rico / USVI
    "AST":   "America/Puerto_Rico",
    "AST-4": "America/Puerto_Rico",

    # Alaska (not in the data but correct for future use)
    "AKST":  "America/Anchorage",
    "AKDT":  "America/Anchorage",
}

def standardize_and_convert_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the dataframe with 'BEGIN_DT' (Naive) and 'CZ_TIMEZONE',
    converts to UTC taking into account DST transitions.
    """
    
    # 1. Clean the CZ_TIMEZONE column
    # Remove extra spaces and map to IANA strings
    df['clean_tz'] = df['CZ_TIMEZONE'].str.upper().str.strip().map(TIMEZONE_MAP)
    
    # Fill unknown timezones with UTC to prevent crashing (or drop them)
    # Warning: You might want to inspect what falls into 'UTC' later
    #df['clean_tz'] = df['clean_tz'].fillna('UTC')

    # 2. Vectorized Conversion
    # We cannot vectorize across mixed timezones, so we group by the timezone.
    utc_chunks = []
    
    # Group by the cleaned timezone string
    for tz_name, group in df.groupby('clean_tz'):
        if tz_name == 'UTC':
            converted = group['BEGIN_DT'].dt.tz_localize('UTC')
        else:
            localized = group['BEGIN_DT'].dt.tz_localize(
                tz_name,
                ambiguous='NaT',      # fall back (clock goes back) -> mark as missing
                nonexistent='shift_forward'  # spring forward -> shift into valid time
            )
            converted = localized.dt.tz_convert('UTC')
            
        # Assign back to the chunk
        group=group.copy()
        group['BEGIN_DT_UTC'] = converted
        utc_chunks.append(group)

    # 3. Reassemble the dataframe
    df_utc = pd.concat(utc_chunks).sort_index()
    
    # Cleanup helper column
    return df_utc.drop(columns=['clean_tz'])

def build_begin_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Build a reliable datetime from BEGIN_YEARMONTH (YYYYMM), BEGIN_DAY, BEGIN_TIME (HHMM).
    Handles missing/odd times by coercing to NaT if needed.
    """
    # Ensure strings with zero-padding
    yyyymm = df["BEGIN_YEARMONTH"].astype("Int64").astype(str)          # e.g., 201001
    day    = df["BEGIN_DAY"].astype("Int64").astype(str).str.zfill(2)   # e.g., 7 -> "07"

    # BEGIN_TIME can be 0, 30, 930, 2359, or missing. Pad to 4 digits.
    time = (
        df["BEGIN_TIME"]
        .astype("Int64")
        .fillna(0)
        .astype(str)
        .str.zfill(4)
    )

    # Concatenate and parse
    dt_str = yyyymm + day + time
    return pd.to_datetime(dt_str, format="%Y%m%d%H%M", errors="coerce")

def load_stormevents_data(
    data_dir,
    pattern="StormEvents_details-ftp_v1.0_d*_c*.csv.gz",
    storm_types=None,
    winter_months=None,
    min_winter_year=1950,
):
    """
    Load and preprocess StormEvents data.

    Parameters:
    -----------
    data_dir : str or Path
        Directory containing StormEvents files
    pattern : str
        File pattern to match
    storm_types : set
        Event types to keep
    winter_months : set
        Months defining winter season
    min_winter_year : int
        Minimum winter season to keep

    Returns:
    --------
    df_scs : pandas.DataFrame
    """

    if storm_types is None:
        storm_types = {"Tornado", "Hail", "Thunderstorm Wind"}

    if winter_months is None:
        winter_months = {12, 1, 2, 3}

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    print(f"Found {len(files)} files")

    dfs = []

    for fp in files:
        m = re.search(r"_d(\d{4})_", fp.name)
        year = int(m.group(1)) if m else None

        print(f"Reading {fp.name} (year={year})")

        df = pd.read_csv(fp, compression="gzip", low_memory=False)

        # Build datetime (your custom function)
        df["BEGIN_DT"] = build_begin_datetime(df)
        df = df.dropna(subset=["BEGIN_DT"])

        # Winter season labeling
        dt = df["BEGIN_DT"]
        df["WINTER_SEASON"] = dt.dt.year - dt.dt.month.isin(winter_months).astype(int)

        # Storm-type filter
        df = df[df["EVENT_TYPE"].isin(storm_types)]

        # Store source year
        df["SOURCE_FILE_YEAR"] = year

        dfs.append(df)

    # Concatenate
    df_scs = pd.concat(dfs, ignore_index=True)

    # Filter by winter season
    df_scs = df_scs[df_scs["WINTER_SEASON"] >= min_winter_year]

    # Convert to UTC (your function)
    df_scs = standardize_and_convert_to_utc(df_scs)

    print("Done.")
    print(df_scs.shape)
    print(df_scs["EVENT_TYPE"].value_counts())
    print(df_scs["WINTER_SEASON"].min(), df_scs["WINTER_SEASON"].max())

    return df_scs

