import cdsapi
import os

client = cdsapi.Client()

dataset = 'reanalysis-era5-single-levels'

variables = {
    'mean_sea_level_pressure': 'msl',
    'geopotential':'z'
}

for year in range(1996,2026):

    for var_name, short_name in variables.items():

        for month in range(1,13):

            month_str = str(month).zfill(2)

            request = {
                'product_type': ['reanalysis'],
                'variable': [var_name],
                'year': [str(year)],
                'month': [month_str],
                'day': [str(d).zfill(2) for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'area': [90, -180, 10, 180] 
            }

            target = (
                f'/data1/lepique/era5_hourly_1996_2025/'
      	        f'era5_{short_name}_{year}_{month_str}.nc'
            )

            if os.path.exists(target):
                print(f'Skipping existing file: {target}')
                continue

            print(f'Downloading {var_name} for {year}-{month_str}...')
            client.retrieve(dataset, request, target)


print('DONE!!!')
