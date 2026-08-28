#era5_daily_climo.py

## Import packages
import xarray as xr
import numpy as np
import calendar
import datetime
from datetime import datetime
import os


## Directory with ERA5 data
#era5_root = '/data1/ERA5/'   ## pressure level data
era5_root = '/data1/ERA5/surface/'  ## surface data

## Years to loop over
years = list(range(1996, 2026))

## Output directory
out_dir = '/data1/ERA5/daily_climo/'


## Function to return day-of-year, SKIPPING Feb 29 on leap years
def doy_no_leaps(dt):
    if dt.month == 2 and dt.day == 29:
        return None
    doy = dt.timetuple().tm_yday
    if calendar.isleap(dt.year) and dt.month >= 3:
        doy = doy - 1
    return doy

## Function to calculate the daily climatology:
def compute_daily_climo(var_short, era5_root, years, out_dir):
    
    # Get grid dimensions from first file 
    first_file = os.path.join(era5_root, var_short, f"{var_short}_{years[0]}.grib")
    
    ds_meta = xr.open_dataset(first_file, engine='cfgrib')
    
    #nlev = len(ds_meta['isobaricInhPa']) ## Include for pressure level data
    
    nlat = len(ds_meta['latitude'])
    
    nlon = len(ds_meta['longitude'])
    
    #levels = ds_meta['isobaricInhPa'].values ## Include for pressure level data
    lats   = ds_meta['latitude'].values
    lons   = ds_meta['longitude'].values
    
    data_var = list(ds_meta.data_vars)[0]
    
    ds_meta.close()

    ## Initialize arrays: (## Add nlev in the dimensions for pressure level data, remove for surface data
    clim_sum_00z     = np.zeros((365, nlat, nlon), dtype=np.float64)
    clim_sum_12z     = np.zeros((365, nlat, nlon), dtype=np.float64)
    clim_count_00z   = np.zeros(365, dtype=np.int32)
    clim_count_12z   = np.zeros(365, dtype=np.int32)
    clim_var_sum_00z = np.zeros((365, nlat, nlon), dtype=np.float64)
    clim_var_sum_12z = np.zeros((365, nlat, nlon), dtype=np.float64)

    # Main loop 
    for y in years:
        fpath = os.path.join(era5_root, var_short, f"{var_short}_{y}.grib")
        print(f"Processing {y}...", flush=True)
        ds = xr.open_dataset(fpath, engine='cfgrib')
    
        for i in range(len(ds.time)):
            date = ds.time[i].values.astype('datetime64[ms]').astype(datetime)
            doy = doy_no_leaps(date)
            if doy is None:
                continue
            idx = doy - 1

            if date.hour == 0:
                clim_sum_00z[idx] +=ds.isel(time=i)[data_var].values
                clim_count_00z[idx] +=1
                
            elif date.hour == 12:
                clim_sum_12z[idx] +=ds.isel(time=i)[data_var].values
                clim_count_12z[idx] +=1

    
        ds.close()

    # Compute mean for 0Z  (3 np.newaxis for pressure level data, 2 for surface data)
    clim_mean_00z = np.where(
    clim_count_00z[:, np.newaxis, np.newaxis] > 0,
    clim_sum_00z / clim_count_00z[:, np.newaxis, np.newaxis],
    np.nan
    )
    ## compute mean for 12Z
    clim_mean_12z = np.where(
    clim_count_12z[:, np.newaxis, np.newaxis] > 0,
    clim_sum_12z / clim_count_12z[:, np.newaxis, np.newaxis],
    np.nan
    )
    
    # second loop to calculate standard deviation (since it requires the mean to calculate)
    for y in years:
        fpath = os.path.join(era5_root, var_short, f"{var_short}_{y}.grib")
        print(f"Processing {y}...", flush=True)
        ds = xr.open_dataset(fpath, engine='cfgrib')
        ds_00z = ds.sel(time=ds.time.dt.hour == 0)
    
        for i in range(len(ds.time)):
            date = ds.time[i].values.astype('datetime64[ms]').astype(datetime)
            doy = doy_no_leaps(date)
            if doy is None:
                continue
            idx = doy - 1

            if date.hour == 0:
                clim_var_sum_00z[idx] += (ds.isel(time=i)[data_var].values - clim_mean_00z[idx]) ** 2

            elif date.hour == 12:
                clim_var_sum_12z[idx] += (ds.isel(time=i)[data_var].values - clim_mean_12z[idx]) ** 2
    
        ds.close()

    # Compute standard deviations (3 np.newaxis for pressure level data)
    clim_std_00z = np.sqrt(clim_var_sum_00z / clim_count_00z[:, np.newaxis, np.newaxis])
    clim_std_12z = np.sqrt(clim_var_sum120z / clim_count_12z[:, np.newaxis, np.newaxis])

    ## Create output path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{var_short}_clim_{min(years)}-{max(years)}.nc")

    ## Write the data out into a .nc file (## add level dim and coord for pressure level data
    ds_out = xr.Dataset(
        {
            f'{var_short}_00z':     xr.DataArray(clim_mean_00z.astype(np.float32), dims=["doy", "latitude", "longitude"], attrs={"long_name": f"Daily climatological mean of {var_short} 00Z"}),
            f'{var_short}_std_00z': xr.DataArray(clim_std_00z.astype(np.float32),  dims=["doy", "latitude", "longitude"], attrs={"long_name": f"Daily climatological std dev of {var_short} 00Z"}),
            f'{var_short}_12z':     xr.DataArray(clim_mean_12z.astype(np.float32), dims=["doy", "latitude", "longitude"], attrs={"long_name": f"Daily climatological mean of {var_short} 12Z"}),
            f'{var_short}_std_12z': xr.DataArray(clim_std_12z.astype(np.float32),  dims=["doy", "latitude", "longitude"], attrs={"long_name": f"Daily climatological std dev of {var_short} 12Z"}),
        },
        coords={
            "doy":       ("doy",       np.arange(1, 366)),
            "latitude":  ("latitude",  lats),
            "longitude": ("longitude", lons),
            "n_years":   ("doy",       clim_count_00z),
        }
    )
    
    ds_out.to_netcdf(out_path, encoding={
        f'{var_short}_00z':     {"zlib": True, "complevel": 4, "dtype": "float32"},
        f'{var_short}_std_00z': {"zlib": True, "complevel": 4, "dtype": "float32"},
        f'{var_short}_12z':     {"zlib": True, "complevel": 4, "dtype": "float32"},
        f'{var_short}_std_12z': {"zlib": True, "complevel": 4, "dtype": "float32"},
    })
    ds_out.close()
    print(f"Saved → {out_path}")
    
        
if __name__ == "__main__":
    
    #variables = ["Z", "T", "U", "V", "O3", "PV", "q", "W"]  ## pressure level variables
    variables = ["slp", "t2m", "d2m", "tcwv"]	 ## surface  variables
    
    for var in variables:
        compute_daily_climo(var, era5_root, years, out_dir)
