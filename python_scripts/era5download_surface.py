## Python script to download era5 surface variables

import cdsapi

##We are downloading this data: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
# Make sure you have a copernicus account (https://accounts.ecmwf.int/auth/realms/ecmwf/protocol/openid-connect/auth)
# You'll need to create the .cdsapirc file in your home directory (you can get the details on how to do this by logging onto the link above and going to your account info

# Define years and variables for downloading
years = list(range(1995,2026))

var_map = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "mean_sea_level_pressure": "slp"
}

variables = list(var_map.keys())

dataset = "reanalysis-era5-single-levels"

client = cdsapi.Client()


for y in years:
    for v_long in variables:
        v_short = var_map[v_long]
        print(f"--- Requestiong Data for Year: {y}, Variable: {v_long} (Short: {v_short}) ---")
        
        # Request info:
        request = {
            "product_type": ["reanalysis"],
            "variable": [v_long],  # Use the long variable name (v_long)
            "year": [str(y)], 
            "month": [
                "01", "02", "03", "04", "05", "06",
                "07", "08", "09", "10", "11", "12"
            ],
            "day": [
                "01", "02", "03", "04", "05", "06",
                "07", "08", "09", "10", "11", "12",
                "13", "14", "15", "16", "17", "18",
                "19", "20", "21", "22", "23", "24",
                "25", "26", "27", "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "12:00"
            ],

            "format": "grib",  # Set to grib format
            "area": [90, -180, 10, 180] 
        }

        # Define the save filename using the short variable name, e.g., ERA5_z_2024.grib
        target = f'./{v_short}/{v_short}_{y}.grib' 
        
       	client = cdsapi.Client()
        client.retrieve(
            dataset,
            request,
            target
        )

print("--- All data requests have been submitted. ---")

        

