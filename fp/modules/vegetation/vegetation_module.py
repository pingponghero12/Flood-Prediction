from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path
import requests
import zipfile
import rasterio
import matplotlib.pyplot as plt
import os

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

def search_products_l2a(token, bbox, date_start, date_end, cloud_thresh=20):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    wkt = f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"

    filter_str = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId' and att/OData.CSC.StringAttribute/Value eq '33UXR') and "
        f"ContentDate/Start gt {date_start} and ContentDate/Start lt {date_end} and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_thresh})"
    )

    params = {"$filter": filter_str, "$orderby": "ContentDate/Start asc", "$top": 5}
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    products = r.json().get("value", [])
    l2a_products = [p for p in products if 'MSIL2A' in p['Name']]
    
    if not l2a_products:
        return None
    return l2a_products[0]

def download_product_if_needed(product, token, out_dir):
    product_name = product['Name']
    zip_path = Path(out_dir) / f"{product_name}.zip"
    if zip_path.exists():
        print(f"{zip_path} exists, skipping download")
        return zip_path
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return zip_path

def extract_bands(zip_file, out_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    
    safe_dir = next(Path(out_dir).glob('*.SAFE'), None)
    if not safe_dir: raise Exception("No .SAFE folder found")
    
    img_folder = list(safe_dir.glob('GRANULE/*/IMG_DATA'))[0]
    r10m_folder = img_folder / 'R10m'
    
    bands_map = {
        'B02': list(r10m_folder.glob('*_B02_10m.jp2'))[0],
        'B03': list(r10m_folder.glob('*_B03_10m.jp2'))[0],
        'B04': list(r10m_folder.glob('*_B04_10m.jp2'))[0],
        'B08': list(r10m_folder.glob('*_B08_10m.jp2'))[0],
    }
    
    bands_data = {}
    profile = None
    
    for name, path in bands_map.items():
        with rasterio.open(path) as src:
            bands_data[name] = src.read(1)
            if profile is None:
                profile = src.profile.copy()
    
    return bands_data, profile

def create_rgb_array(bands):
    rgb = np.stack([bands['B04'], bands['B03'], bands['B02']], axis=0).astype(float)
    # Normalize for visualization/integration
    # Simple clip 0-3000 reflectance
    rgb = np.clip(rgb / 3000.0, 0, 1)
    return rgb

def get_vegetation_map(coords: Dict[str, float], time: str, is_single_run: bool) -> Dict[str, Any]:
    module_dir = Path(__file__).parent
    out_dir_path = module_dir / "data_vegetation"
    out_dir_path.mkdir(exist_ok=True)
    
    ndvi_geotiff_path = out_dir_path / "ndvi_geotiff.tif"

    # If cache exists, read metadata and data
    if ndvi_geotiff_path.exists():
        print("Using cached vegetation data...")
        with rasterio.open(ndvi_geotiff_path) as src:
            ndvi = src.read(1)
            profile = src.profile.copy()
        
        # Try to recover RGB if possible, else return None for RGB
        # (For this example, we assume we want to download if we need fresh data, 
        # but here we just return NDVI if cached to be fast)
        return {
            "ndvi": ndvi,
            "rgb": None, # RGB heavy to cache as numpy array, usually re-generated
            "profile": profile
        }

    # Download Flow
    from getpass import getpass
    print("Cached vegetation not found. Downloading Sentinel-2...")
    username = input("CDSE Username: ")
    password = getpass("Password: ")
    
    lat0, lon0 = 50.433, 16.653
    delta = 0.45
    bbox = [lon0 - delta, lat0 - delta, lon0 + delta, lat0 + delta]
    date_start = "2025-06-15T00:00:00.000Z"
    date_end = "2025-07-15T23:59:59.999Z"
    
    token = get_access_token(username, password)
    prod = search_products_l2a(token, bbox, date_start, date_end)
    
    if not prod:
        return {"ndvi": None, "profile": None}
    
    zip_path = download_product_if_needed(prod, token, out_dir_path)
    bands, profile = extract_bands(zip_path, out_dir_path)
    
    ndvi = (bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'] + 1e-6)
    rgb = create_rgb_array(bands)
    
    # Save NDVI
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(ndvi_geotiff_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    # Save RGB Preview
    plt.imsave(out_dir_path / "rgb_image.png", np.moveaxis(rgb, 0, -1))
    
    return {
        "ndvi": ndvi,
        "rgb": rgb, # Shape (3, H, W)
        "profile": profile
    }
