from typing import Dict, Any
import os
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from whitebox import WhiteboxTools
from pathlib import Path
import requests
import zipfile
import shutil
from getpass import getpass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data_topography")

DEFAULT_FILES = {
    "dem_filled": "dem_filled.tif",
    "flow_dir": "flow_dir.tif",
    "flow_acc": "flow_acc.tif",
    "slope": "slope.tif",
    "twi": "twi.tif",
}

def get_access_token(username, password):
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Login failed: {response.status_code} - {response.text}")

def search_dem_product(token, bbox):
    """Search for DEM products in Copernicus (COP-DEM)."""
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    wkt = f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
    
    filter_str = (
        f"Collection/Name eq 'COP-DEM' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )
    
    params = {
        "$filter": filter_str,
        "$top": 50 
    }
    
    print(f"Querying OData for COP-DEM...")
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    
    products = r.json().get("value", [])
    
    if not products:
        raise Exception(f"CRITICAL: No DEM data found for bbox {bbox}. Check coordinates.")

    # Priority Filtering:
    # 1. COG_10 (30m Cloud Optimized)
    # 2. _30_ (30m SAR/Raw)
    # 3. COG_30 (90m Cloud Optimized)
    
    cog_30 = [p for p in products if "COG_10" in p["Name"]]
    any_30 = [p for p in products if "_30_" in p["Name"] or "10_" in p["Name"]]
    
    if cog_30:
        print(f"Selected High-Res COG DEM: {cog_30[0]['Name']}")
        return cog_30[0]
    
    if any_30:
        print(f"Selected 30m Resolution DEM: {any_30[0]['Name']}")
        return any_30[0]
        
    print("Warning: No 30m data found. Falling back to 90m.")
    return products[0]

def download_dem_if_needed(product, token, out_dir):
    product_name = product['Name']
    zip_path = Path(out_dir) / f"{product_name}.zip"
    
    if zip_path.exists():
        print(f"{zip_path} exists, skipping download")
        return zip_path
        
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Downloading {product_name}...")
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return zip_path

def extract_dem(zip_file, out_dir):
    """Extract DEM from downloaded product."""
    print(f"Extracting {zip_file}...")
    extract_path = Path(out_dir) / zip_file.stem
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    extensions = {".tif", ".TIF", ".dt2", ".DT2", ".dem", ".DEM"}
    all_files = list(extract_path.rglob("*"))
    
    candidate_files = [
        f for f in all_files 
        if f.suffix in extensions and "CoV" not in f.name
    ]
    
    if not candidate_files:
        # Fallback for files without standard extension
        large_files = [f for f in all_files if f.is_file() and f.stat().st_size > 5000000] # > 5MB
        if large_files:
             dem_file = max(large_files, key=lambda f: f.stat().st_size)
             print(f"No standard extension found. Guessing largest file is DEM: {dem_file.name}")
             return dem_file
        raise Exception("No valid DEM file found in extracted data.")
    
    dem_file = max(candidate_files, key=lambda f: f.stat().st_size)
    print(f"Found source DEM file: {dem_file.name}")
    return dem_file

def convert_to_geotiff(src_path, dst_path):
    """
    Reads the source raster and writes a clean, standard 256x256 tiled GeoTIFF.
    Fixes incompatibility between DTED scanlines and GeoTIFF tiles.
    """
    print(f"Converting {src_path.name} to standard GeoTIFF...")
    with rasterio.open(src_path) as src:
        # Start with a clean profile
        profile = src.profile.copy()
        
        # FIX: Enforce standard block sizes to prevent "Bad value 1 for TileWidth"
        profile.update(
            driver='GTiff',
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        data = src.read(1).astype(rasterio.float32)
        
        # Handle nodata
        nodata_val = src.nodata
        if nodata_val is None:
            # DTED often uses -32767
            if np.nanmin(data) < -32000:
                nodata_val = -32767

        if nodata_val is not None:
             data[data == nodata_val] = np.nan
        
        # Write clean TIF
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data, 1)
            
    print(f"Conversion complete: {dst_path}")

def get_or_download_dem():
    data_dir = Path(DEFAULT_DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    dem_tif_path = data_dir / "dem.tif"
    
    if dem_tif_path.exists():
        print("Using cached DEM data...")
        return str(dem_tif_path)
    
    print("Cached DEM not found. Downloading from Copernicus...")
    username = input("CDSE Username: ")
    password = getpass("Password: ")
    
    lat0, lon0 = 50.433, 16.653
    delta = 0.05
    bbox = [lon0 - delta, lat0 - delta, lon0 + delta, lat0 + delta]
    
    token = get_access_token(username, password)
    prod = search_dem_product(token, bbox)
    zip_path = download_dem_if_needed(prod, token, data_dir)
    extracted_dem = extract_dem(zip_path, data_dir)
    
    convert_to_geotiff(extracted_dem, dem_tif_path)
    
    return str(dem_tif_path)

def _read_raster_maybe(path: str):
    try:
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return None

def _run_whitebox(dem_path: str, out_dir: str):
    wbt = WhiteboxTools()
    wbt.work_dir = out_dir
    wbt.verbose = False

    dem_filled = os.path.join(out_dir, DEFAULT_FILES["dem_filled"])
    flow_dir = os.path.join(out_dir, DEFAULT_FILES["flow_dir"])
    flow_acc = os.path.join(out_dir, DEFAULT_FILES["flow_acc"])
    slope = os.path.join(out_dir, DEFAULT_FILES["slope"])
    twi = os.path.join(out_dir, DEFAULT_FILES["twi"])

    print("1. Filling Depressions...")
    wbt.fill_depressions(dem=dem_path, output=dem_filled)

    print("2. Calculating Flow Direction...")
    wbt.d8_pointer(dem=dem_filled, output=flow_dir)

    print("3. Calculating Flow Accumulation...")
    wbt.d8_flow_accumulation(i=dem_filled, output=flow_acc, out_type="cells")

    print("4. Calculating Slope...")
    wbt.slope(dem=dem_filled, output=slope, zfactor=None)

    print("5. Calculating TWI manually...")
    try:
        if not os.path.exists(flow_acc) or not os.path.exists(slope):
            # Fallback if WBT failed silently
            raise Exception("WBT outputs missing.")
            
        with rasterio.open(flow_acc) as acc_src:
            acc = acc_src.read(1)
            cellsize = acc_src.res[0]

        with rasterio.open(slope) as sl_src:
            slope_deg = sl_src.read(1)

        slope_rad = np.radians(slope_deg)
        slope_rad[slope_rad < 1e-5] = 1e-5 

        sca = acc * cellsize
        twi_arr = np.log(sca / np.tan(slope_rad) + 1e-6)

        with rasterio.open(dem_path) as src_dem:
            # Clean profile for TWI output
            twi_profile = src_dem.profile.copy()
            twi_profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(twi, "w", **twi_profile) as dst:
            dst.write(twi_arr.astype(np.float32), 1)
            
    except Exception as e: 
        print(f"ERROR calculating TWI: {e}")
        twi_arr = np.full((256, 256), np.nan)

def topographic_module(
    dem_path: str = None,
    out_dir: str = DEFAULT_OUT_DIR,
    visualize: bool = True
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    if dem_path is None:
        dem_path = get_or_download_dem()

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM file does not exist: {dem_path}")

    files_missing = any(not os.path.exists(os.path.join(out_dir, f)) for f in DEFAULT_FILES.values())

    if files_missing:
        print(f"Running topographic analysis in: {out_dir}")
        _run_whitebox(dem_path, out_dir)
    else:
        print("Using cached topographic analysis results...")

    data: Dict[str, Any] = {}
    for key, fname in DEFAULT_FILES.items():
        path = os.path.join(out_dir, fname)
        arr = _read_raster_maybe(path)
        data[key] = {"array": arr, "path": path}

    if visualize:
        try:
            from fp.modules.data_visualization.data_visualization_module import (
                visualize_topography_data_separate,
            )
            visualize_topography_data_separate(
                {k: v["array"] for k, v in data.items()},
                out_dir=out_dir
            )
        except ImportError:
            print("Visualization skipped (module not found).")

    return {k: v["array"] for k, v in data.items()}

if __name__ == "__main__":
    topo = topographic_module(visualize=True)
    print("Topography maps created:", list(topo.keys()))
