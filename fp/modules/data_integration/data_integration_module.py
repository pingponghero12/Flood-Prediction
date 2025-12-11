from typing import Dict, Any
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import matplotlib.pyplot as plt
import os

def integrate_data(
    topography: Dict[str, Any], 
    vegetation: Dict[str, Any], 
    meteorology: Dict[str, Any], 
    out_dir: str,
    target_resolution: float = 30.0
) -> Dict[str, Any]:
    """
    Integrates Vegetation and Topography by cropping both to their
    Common Spatial Subset (Intersection) at 30m resolution.
    """
    print("Starting Spatial Integration (Common Subset)...")
    os.makedirs(out_dir, exist_ok=True)

    veg_profile = vegetation.get("profile")
    topo_profile = topography.get("profile")
    
    if not veg_profile or not topo_profile:
        print("CRITICAL: Missing spatial metadata. Cannot align.")
        return {}

    # --- Step 1: Calculate the Intersection (Common Bounding Box) ---
    
    # Get bounds of Vegetation (already in its own CRS, usually UTM)
    v_height, v_width = veg_profile['height'], veg_profile['width']
    v_left, v_bottom, v_right, v_top = rasterio.transform.array_bounds(
        v_height, v_width, veg_profile['transform']
    )
    
    # Get bounds of Topography (in its own CRS, usually Lat/Lon)
    t_height, t_width = topo_profile['height'], topo_profile['width']
    t_left, t_bottom, t_right, t_top = rasterio.transform.array_bounds(
        t_height, t_width, topo_profile['transform']
    )
    
    # We choose Vegetation CRS (UTM) as the Target CRS
    dst_crs = veg_profile['crs']
    
    # Reproject Topo bounds into Veg CRS to check overlap
    t_left_proj, t_bottom_proj, t_right_proj, t_top_proj = transform_bounds(
        topo_profile['crs'], dst_crs, t_left, t_bottom, t_right, t_top
    )
    
    # Calculate Intersection Coordinates
    inter_left = max(v_left, t_left_proj)
    inter_bottom = max(v_bottom, t_bottom_proj)
    inter_right = min(v_right, t_right_proj)
    inter_top = min(v_top, t_top_proj)
    
    if inter_left >= inter_right or inter_bottom >= inter_top:
        print("ERROR: Datasets do not overlap!")
        return {}

    print(f"Intersection Found: {inter_left:.2f}, {inter_bottom:.2f}, {inter_right:.2f}, {inter_top:.2f}")

    # --- Step 2: Define New Grid ---
    dst_width = int((inter_right - inter_left) / target_resolution)
    dst_height = int((inter_top - inter_bottom) / target_resolution)
    
    dst_transform = rasterio.transform.from_origin(
        inter_left, inter_top, target_resolution, target_resolution
    )
    
    print(f"Target Common Grid: 30m Res, Size: {dst_width}x{dst_height}")
    
    integrated_data = {}

    # --- Step 3: Reproject Both ---
    
    # Topography
    print("Cropping & Reprojecting Topography...")
    for key, arr in topography.items():
        if key == "profile" or arr is None: continue
        
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=topo_profile['transform'],
            src_crs=topo_profile['crs'],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        integrated_data[f"topo_{key}"] = dst_arr

    # Vegetation
    print("Cropping & Reprojecting Vegetation...")
    if vegetation.get("ndvi") is not None:
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=vegetation["ndvi"],
            destination=dst_arr,
            src_transform=veg_profile['transform'],
            src_crs=veg_profile['crs'],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        integrated_data["veg_ndvi"] = dst_arr

    if vegetation.get("rgb") is not None:
        src_rgb = vegetation["rgb"]
        dst_rgb = np.zeros((3, dst_height, dst_width), dtype=np.float32)
        for i in range(3):
            reproject(
                source=src_rgb[i],
                destination=dst_rgb[i],
                src_transform=veg_profile['transform'],
                src_crs=veg_profile['crs'],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average
            )
        integrated_data["veg_rgb"] = dst_rgb

    integrated_data["meteorology"] = meteorology
    
    # --- Step 4: Visualize ---
    visualize_integration(integrated_data, out_dir)
    
    return integrated_data

def visualize_integration(data, out_dir):
    print("Generating integrated comparison image...")
    
    dem = data.get("topo_dem_filled")
    rgb = data.get("veg_rgb")
    ndvi = data.get("veg_ndvi")
    
    if dem is None:
        print("Error: DEM data missing.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot DEM
    im = ax1.imshow(dem, cmap='terrain')
    ax1.set_title("Topography (Common Subset)")
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot Vegetation
    if rgb is not None:
        rgb_plot = np.moveaxis(rgb, 0, -1)
        rgb_plot = np.clip(rgb_plot, 0, 1)
        ax2.imshow(rgb_plot)
        ax2.set_title("Vegetation (True Color - Common Subset)")
    elif ndvi is not None:
        # FIXED: Set vmin=0 to ignore water/negative values in visualization
        im2 = ax2.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title("Vegetation (NDVI - Common Subset)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "Missing Vegetation", ha='center')
        ax2.set_title("Vegetation Missing")
        
    ax2.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "integrated_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Comparison saved to: {save_path}")
    
    try:
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)
    except:
        plt.close(fig)
