#TE_processing.py

import pandas as pd

def parse_nodefile(filepath):
    """ 
    Parse a TempestExtremes StitchNodes nodefile into a single dataframe with one row per ETC center detection, tagged with a unique track_id
    """
    records=[]
    track_id = 0 # 1 is the first "start" line

    with open(filepath,'r') as f:
        for line in f:
            parts = line.split()
            if len(parts)==0:
                continue
            if parts[0] == 'start':
                ## new trajectory tarting; increment track_id
                track_id += 1
            else:
                ## data line: parse fields and append to records list tagged with current track_id
                records.append({
                    'track_id': track_id,
                    'lon': float(parts[2]),
                    'lat': float(parts[3]),
                    'msl': float(parts[4]) / 100.0,  # convert Pa to hPa
                    'z': float(parts[5]),
                    'year': int(parts[6]),
                    'month': int(parts[7]),
                    'day': int(parts[8]),
                    'hour': int(parts[9]),
                })
    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    return df


# US domain bounds (as in Kiefer et al. 2026)
# 135W-55W = 225E-305E in 0-360 convention
LON_MIN = 225.0
LON_MAX = 305.0
LAT_MIN = 20.0
LAT_MAX = 70.0


def filter_tracks(all_tracks, seasons=(1996, 2025)):
    """
    Return a dataframe containing only cyclone tracks that
      1. pass through the CONUS domain at least once,
      2. have at least one time step in the DJFM months (Dec, Jan, Feb, Mar), and
      3. belong to a DJFM winter season within `seasons` (inclusive).

    DJFM winter-season convention: December is attached to the FOLLOWING year's
    JFM (e.g. Dec 2010 -> season 2011). Every DJFM node of a given track maps to
    the same season -- including a track that runs from December across the
    Dec 31 -> Jan 1 boundary into January, which stays a single track -- so the
    season window keeps year-boundary-crossing tracks whole and only removes the
    incomplete end seasons (a JFM-only first season with no preceding December,
    or a December-only final season with no following JFM).

    Pass seasons=None to disable the season window (month filter only).
    """

    valid_track_ids = []

    for track_id, track_df in all_tracks.groupby('track_id'):

        in_domain = (
            (track_df['lon'] >= LON_MIN) &
            (track_df['lon'] <= LON_MAX) &
            (track_df['lat'] >= LAT_MIN) &
            (track_df['lat'] <= LAT_MAX)
        )

        if in_domain.any():
            valid_track_ids.append(track_id)

    us_tracks = all_tracks[
        all_tracks['track_id'].isin(valid_track_ids)
    ]

    us_tracks_djfm = us_tracks[
        us_tracks['month'].isin([12,1,2,3])
    ]

    if seasons is not None:
        ## December -> following year's JFM season; keeps Dec->Jan tracks intact
        season = us_tracks_djfm['year'] + (us_tracks_djfm['month'] == 12).astype(int)
        lo, hi = seasons
        us_tracks_djfm = us_tracks_djfm[(season >= lo) & (season <= hi)]

    return us_tracks_djfm

