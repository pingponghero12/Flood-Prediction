from typing import Dict, Any, Optional
import os
import requests
import xarray as xr
import rioxarray as rxr
import numpy as np
import rasterio
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data_meteorology")

def get_latest_gfs_url(bbox: Dict[str, float]) -> str:
    """
    Constructs the URL to download the latest available GFS forecast from NOAA.
    We use the NOMADS filter to download ONLY the Kłodzko region and ONLY rain/moisture variables.
    This requires NO LOGIN.
    """
    # 1. Determine date and cycle (GFS updates every 6 hours: 00, 06, 12, 18 UTC)
    # We look back a few hours to ensure the data is fully uploaded to the server
    now = datetime.now(timezone.utc)
    
    # Simple logic to find the latest likely available cycle
    # (Data usually takes ~4 hours to publish)
    target_time = now - timedelta(hours=4)
    date_str = target_time.strftime("%Y%m%d")
    hour = target_time.hour
    
    if hour < 6: cycle = "00"
    elif hour < 12: cycle = "06"
    elif hour < 18: cycle = "12"
    else: cycle = "18"

    print(f"Targeting GFS Cycle: {date_str} {cycle}z")

    # 2. Define URL parameters for NOAA NOMADS Filter
    # We want the forecast for +24 hours (f024) to see accumulation
    # Variables: APCP (Total Precipitation), SOILW (Volumetric Soil Moisture)
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    
    file_param = f"gfs.t{cycle}z.pgrb2.0p25.f024" # f024 = 24 hour forecast
    dir_param = f"/gfs.{date_str}/{cycle}/atmos"
    
    # Bounding Box (NOAA requires slightly wider margins)
    # Note: Kłodzko is approx Lat 50.4, Lon 16.6
    left = bbox['minx'] - 0.5
    right = bbox['maxx'] + 0.5
    top = bbox['maxy'] + 0.5
    bottom = bbox['miny'] - 0.5

    # Construct full query
    # var_APCP=on  -> Accumulated Precipitation
    # var_SOILW=on -> Soil Moisture
    # lev_surface=on -> Surface level (for rain)
    # lev_0-0.1_m_below_ground=on -> Top soil layer (for moisture)
    query_url = (
        f"{base_url}?file={file_param}"
        f"&lev_surface=on&lev_0-0.1_m_below_ground=on"
        f"&var_APCP=on&var_SOILW=on"
        f"&subregion=&leftlon={left}&rightlon={right}&toplat={top}&bottomlat={bottom}"
        f"&dir={dir_param}"
    )
    
    return query_url

def download_gfs_data(url: str, out_path: Path):
    """Downloads the GRIB file from NOAA."""
    if out_path.exists():
        # Check if file is fresh (less than 6 hours old)
        file_age = datetime.now().timestamp() - out_path.stat().st_mtime
        if file_age < 3600 * 6:
            print("Using cached GFS data (fresh enough)...")
            return

    print(f"Downloading GFS Forecast from NOAA...")
    print(f"URL: {url}")
    
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete: {out_path}")
        else:
            print(f"Failed to download GFS. Status: {r.status_code}")
            print(r.text)
            raise Exception("NOAA NOMADS download failed.")
    except Exception as e:
        print(f"Connection error: {e}")
        raise

def process_grib_to_geotiff(grib_path: Path, out_dir: Path, bounds: Dict[str, float]):
    """
    Reads the GRIB file using xarray/cfgrib and converts to GeoTIFF.
    """
    rain_path = out_dir / "precip.tif"
    moisture_path = out_dir / "soil_moisture.tif"
    
    # Return paths if they exist and we want to skip processing? 
    # Better to re-process to ensure consistency with bounds.
    
    print("Processing GRIB data to GeoTIFF...")
    
    try:
        # Load GRIB file
        # Note: cfgrib might generate .idx files alongside the .grib2 file
        
        # 1. Precipitation (APCP)
        # GFS variable often named 'tp' (total precipitation) or 'apcp' in xarray
        # filter_by_keys is safer
        ds_rain = xr.open_dataset(
            grib_path, 
            engine="cfgrib", 
            backend_kwargs={'filter_by_keys': {'shortName': 'tp', 'typeOfLevel': 'surface'}}
        )
        
        # 2. Soil Moisture (SOILW)
        # GFS variable often named 'soilw' or 'swvl1' (Volumetric soil moisture layer 1)
        ds_soil = xr.open_dataset(
            grib_path, 
            engine="cfgrib", 
            backend_kwargs={'filter_by_keys': {'shortName': 'soilw', 'typeOfLevel': 'depthBelowLandLayer'}}
        )

        # Helper to clip and save
        def save_var(ds, var_name_in_ds, out_file):
            da = ds[var_name_in_ds]
            
            # Ensure CRS is written (GFS is usually WGS84 / EPSG:4326)
            da.rio.write_crs("epsg:4326", inplace=True)
            
            # Clip to Kłodzko bounds
            da_clipped = da.rio.clip_box(
                minx=bounds['minx'], miny=bounds['miny'], 
                maxx=bounds['maxx'], maxy=bounds['maxy']
            )
            
            # Write to TIF
            da_clipped.rio.to_raster(out_file, driver="GTiff", compress='LZW')
            print(f"Saved {out_file}")
            
            # Return array and profile for immediate use
            with rasterio.open(out_file) as src:
                return src.read(1), src.profile

        # Get the actual variable names from the dataset (they can vary by GFS version)
        # We take the first data variable available
        rain_var = list(ds_rain.data_vars)[0]
        soil_var = list(ds_soil.data_vars)[0]
        
        rain_arr, profile = save_var(ds_rain, rain_var, rain_path)
        soil_arr, _ = save_var(ds_soil, soil_var, moisture_path)
        
        return {
            "precip": rain_arr,       # Unit: kg/m^2 (mm)
            "soil_moisture": soil_arr, # Unit: Fraction (0-1) or kg/m^2 depending on layer
            "profile": profile
        }

    except ImportError:
        print("CRITICAL ERROR: 'cfgrib' or 'eccodes' not installed.")
        print("Please install: conda install -c conda-forge eccodes cfgrib")
        return {}
    except Exception as e:
        print(f"Error processing GRIB: {e}")
        return {}

def meteorological_module(coords: Dict[str, float] = None, time: str = "") -> Dict[str, Any]:
    """
    Main entry point for meteorological data.
    Downloads forecast and returns precipitation and moisture maps.
    """
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    
    # Define Bounds for Kłodzko (if not provided in coords)
    # Defaulting to Kłodzko Valley roughly
    if coords:
        # Assuming coords is a center point, create a box
        lat, lon = coords.get('lat', 50.4), coords.get('lon', 16.6)
        delta = 0.5
        bounds = {
            'minx': lon - delta, 'miny': lat - delta,
            'maxx': lon + delta, 'maxy': lat + delta
        }
    else:
        # Hardcoded Kłodzko bounds from your draft
        bounds = {'minx': 16.0, 'miny': 49.9, 'maxx': 17.5, 'maxy': 50.7}

    grib_path = Path(DEFAULT_DATA_DIR) / "forecast.grib2"
    
    # 1. Get URL and Download
    try:
        url = get_latest_gfs_url(bounds)
        download_gfs_data(url, grib_path)
    except Exception as e:
        print(f"Warning: Could not download fresh weather data ({e}). Trying cache...")
        if not grib_path.exists():
            print("No cached weather data available.")
            return {}

    # 2. Process to GeoTIFF and return arrays
    data = process_grib_to_geotiff(grib_path, Path(DEFAULT_DATA_DIR), bounds)
    
    return data

if __name__ == "__main__":
    # Test run
    met_data = meteorological_module()
    print("Meteorological Data Keys:", met_data.keys())
