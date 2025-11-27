import requests
import zipfile
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.crs import CRS

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
    """Search for L2A products in bbox [(minlon,minlat,maxlon,maxlat)] and date range."""
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    wkt = f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
    params = {
        "$filter": f"Collection/Name eq 'SENTINEL-2' and "
                   f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
                   f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
                   f"ContentDate/Start gt {date_start} and ContentDate/Start lt {date_end} and "
                   f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_thresh})",
        "$orderby": "ContentDate/Start asc",
        "$top": 5
    }
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
    print(r)
    r.raise_for_status()
    products = r.json().get("value", [])
    l2a_products = [p for p in products if 'MSIL2A' in p['Name']]
    assert l2a_products, "No L2A products found"
    print("diala")
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
    
    safe_dir = None
    for item in Path(out_dir).iterdir():
        if item.is_dir() and item.name.endswith('.SAFE'):
            safe_dir = item
            break
    
    if not safe_dir:
        raise Exception("No .SAFE folder found")
    
    img_folder = list(safe_dir.glob('GRANULE/*/IMG_DATA'))
    
    if not img_folder:
        raise Exception("No IMG_DATA folder found")
    
    img_folder = img_folder[0]
    
    b02 = list(img_folder.glob('*_B02.jp2'))[0]
    b03 = list(img_folder.glob('*_B03.jp2'))[0]
    b04 = list(img_folder.glob('*_B04.jp2'))[0]
    b08 = list(img_folder.glob('*_B08.jp2'))[0]
    
    bands_data = {}
    transform = None
    crs = None
    
    for band_name, band_path in [('B02', b02), ('B03', b03), ('B04', b04), ('B08', b08)]:
        with rasterio.open(band_path) as src:
            bands_data[band_name] = src.read(1)
            if transform is None:
                transform = src.transform
                crs = src.crs
    
    return bands_data, transform, crs, safe_dir

def create_ndvi_geotiff(bands, transform, crs, out_path):
    ndvi = (bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'] + 1e-6)
    with rasterio.open(
        out_path, 'w', driver='GTiff', height=ndvi.shape[0], width=ndvi.shape[1],
        count=1, dtype=ndvi.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(ndvi, 1)

def create_rgb_geotiff(bands, transform, crs, out_path):
    rgb = np.stack([bands['B04'], bands['B03'], bands['B02']], axis=0)
    with rasterio.open(
        out_path, 'w', driver='GTiff', height=rgb.shape[1], width=rgb.shape[2],
        count=3, dtype=rgb.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(rgb)

def visualize_geotiff_rgb(rgb_path, fig_path):
    with rasterio.open(rgb_path) as src:
        rgb = src.read()
    # Normalize for display
    rgb_display = np.dstack([rgb[0], rgb[1], rgb[2]]) / 10000.0
    rgb_display = np.clip(rgb_display * 3.5, 0, 1)  # Adjusted for better contrast
    rgb_display = np.power(rgb_display, 1/2.2)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_display)
    plt.title('Sentinel-2 RGB Composite (L2A)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data_vegetation")
    args = parser.parse_args()
    
    # Bbox for Kłodzko, ~100x100km zone (approx ±0.45° lat/lon)
    lat0, lon0 = 50.433, 16.653
    delta = 0.45
    bbox = [lon0 - delta, lat0 - delta, lon0 + delta, lat0 + delta]
    print(f"Looking in bbox: {bbox} (minlon, minlat, maxlon, maxlat)")
    print("View in OpenStreetMap: Search for these coordinates to see the region.")
    
    date_start = "2025-06-15T00:00:00.000Z"
    date_end = "2025-07-15T23:59:59.999Z"
    
    username = input("CDSE Username: ")
    password = getpass("Password: ")
    
    token = get_access_token(username, password)
    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(exist_ok=True)
    
    print("Searching for L2A...")
    prod = search_products_l2a(token, bbox, date_start, date_end)
    zip_path = download_product_if_needed(prod, token, out_dir_path)
    
    print("Extracting bands...")
    bands, transform, crs, _ = extract_bands(zip_path, out_dir_path)
    
    ndvi_path = out_dir_path / "ndvi_geotiff.tif"
    rgb_path = out_dir_path / "rgb_geotiff.tif"
    create_ndvi_geotiff(bands, transform, crs, ndvi_path)
    create_rgb_geotiff(bands, transform, crs, rgb_path)
    print("NDVI and RGB GeoTIFFs saved.")
    
    fig_path = out_dir_path / "rgb_image.png"
    visualize_geotiff_rgb(rgb_path, fig_path)
    print(f"Done! RGB image saved to {fig_path}")

if __name__ == "__main__":
    from getpass import getpass
    main()
