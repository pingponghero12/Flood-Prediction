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
    Integrates Vegetation, Topography, and Meteorology data.
    Aligns all layers to the Common Spatial Subset (Intersection) at 30m resolution.
    """
    print("Starting Spatial Integration (Common Subset)...")
    os.makedirs(out_dir, exist_ok=True)

    veg_profile = vegetation.get("profile")
    topo_profile = topography.get("profile")
    
    # Meteorology might be missing if download failed, but we can still proceed
    met_profile = meteorology.get("profile")
    
    if not veg_profile or not topo_profile:
        print("CRITICAL: Missing spatial metadata for Topo or Veg. Cannot align.")
        return {}

    # --- Step 1: Calculate the Intersection (Common Bounding Box) ---
    
    # Get bounds of Vegetation (Target CRS)
    v_height, v_width = veg_profile['height'], veg_profile['width']
    v_left, v_bottom, v_right, v_top = rasterio.transform.array_bounds(
        v_height, v_width, veg_profile['transform']
    )
    
    # Get bounds of Topography
    t_height, t_width = topo_profile['height'], topo_profile['width']
    t_left, t_bottom, t_right, t_top = rasterio.transform.array_bounds(
        t_height, t_width, topo_profile['transform']
    )
    
    dst_crs = veg_profile['crs']
    
    # Transform Topo bounds to Veg CRS
    t_left_proj, t_bottom_proj, t_right_proj, t_top_proj = transform_bounds(
        topo_profile['crs'], dst_crs, t_left, t_bottom, t_right, t_top
    )
    
    # Calculate Intersection (Topo vs Veg)
    # Note: We prioritize Topo/Veg overlap. If weather is smaller, we'll fill with edge values.
    # If weather is larger, we crop.
    inter_left = max(v_left, t_left_proj)
    inter_bottom = max(v_bottom, t_bottom_proj)
    inter_right = min(v_right, t_right_proj)
    inter_top = min(v_top, t_top_proj)
    
    if inter_left >= inter_right or inter_bottom >= inter_top:
        print("ERROR: Topography and Vegetation datasets do not overlap!")
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

    # --- Step 3: Reproject Layers ---
    
    # A. Topography
    print("Reprojecting Topography...")
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

    # B. Vegetation
    print("Reprojecting Vegetation...")
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

    # C. Meteorology (Rain)
    if meteorology.get("precip") is not None and met_profile:
        print("Reprojecting Meteorology (Rain)...")
        # Use bilinear resampling to smooth the coarse weather grid
        # Use 'nearest' if you want blocky pixels showing the raw model grid
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
        
        reproject(
            source=meteorology["precip"],
            destination=dst_arr,
            src_transform=met_profile['transform'],
            src_crs=met_profile['crs'],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        integrated_data["met_precip"] = dst_arr
    else:
        print("Warning: Precip data missing or no profile. Skipping Rain integration.")

    # --- Step 4: Visualize ---
    visualize_integration(integrated_data, out_dir)
    
    # Return everything including metadata for prediction module
    integrated_data['transform'] = dst_transform
    integrated_data['crs'] = dst_crs
    
    return integrated_data

def visualize_integration(data, out_dir):
    print("Generating integrated comparison image (3-Panel)...")
    
    dem = data.get("topo_dem_filled")
    rgb = data.get("veg_rgb")
    ndvi = data.get("veg_ndvi")
    rain = data.get("met_precip")
    
    if dem is None:
        print("Error: DEM data missing.")
        return

    # Create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. Topography
    im1 = ax1.imshow(dem, cmap='terrain')
    ax1.set_title("Topography (DEM)")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Elevation (m)")
    
    # 2. Vegetation
    if rgb is not None:
        rgb_plot = np.moveaxis(rgb, 0, -1)
        rgb_plot = np.clip(rgb_plot, 0, 1)
        ax2.imshow(rgb_plot)
        ax2.set_title("Vegetation (True Color)")
    elif ndvi is not None:
        im2 = ax2.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title("Vegetation (NDVI)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="NDVI")
    else:
        ax2.text(0.5, 0.5, "Missing", ha='center')
        ax2.set_title("Vegetation Missing")
    ax2.axis('off')

    # 3. Rainfall
    if rain is not None:
        # Use 'Blues' or 'nipy_spectral' for rain
        # Set minimal transparency for 0 rain if needed, but standard imshow is fine
        im3 = ax3.imshow(rain, cmap='Blues', alpha=0.8)
        ax3.set_title("Forecast Rainfall (24h)")
        
        # Determine max for colorbar scaling dynamically or fixed
        vmax = np.nanmax(rain) if np.nanmax(rain) > 0 else 10
        im3.set_clim(0, vmax)
        
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Precipitation (mm)")
    else:
        ax3.text(0.5, 0.5, "Missing Rain Data", ha='center')
        ax3.set_title("Rainfall Missing")
    ax3.axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, "integrated_comparison_3panel.png")
    plt.savefig(save_path, dpi=150)
    print(f"Comparison saved to: {save_path}")
    
    try:
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)
    except:
        plt.close(fig)
