from typing import Dict, Any
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _safe_maybe_log(arr):
    if arr is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.array(arr, dtype=float)
        # Avoid log(0) warnings by setting <= 0 to NaN
        out[out <= 0] = np.nan
        return np.log(out)
    
def _safe_maybe_log2(arr):
    if arr is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.array(arr, dtype=float)
        out[out <= 0] = np.nan
        return np.log2(out)

def visualize_topography_data_separate(data: Dict[str, Any], out_dir: str = "outputs") -> None:
    """Visualizes topography arrays (DEM, Slope, TWI, etc.) and saves to PNG."""
    os.makedirs(out_dir, exist_ok=True)
    mapping_titles = {
        "dem_filled": "1. DEM (filled)",
        "flow_dir": "2. Flow Direction (D8)",
        "flow_acc": "3. Flow Accumulation",
        "slope": "4. Slope (deg)",
        "twi": "5. Topographic Wetness Index (TWI)"
    }

    for key, arr in data.items():
        title = mapping_titles.get(key, key)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Check for empty or None data
        if arr is None or np.all(np.isnan(arr)):
            ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center", fontsize=14, color="red")
            ax.axis("off")
            out_path = os.path.join(out_dir, f"{key}_missing.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            continue

        plot_arr = arr.copy()
        
        # Log-scale Flow Accumulation for better visibility
        if key == "flow_acc":
            plot_arr = _safe_maybe_log(plot_arr)

        # Log2-scale Flow Direction (powers of 2: 1, 2, 4, 8, 16...)
        if key == "flow_dir":
            plot_arr = _safe_maybe_log2(plot_arr)

        # Choose colormap based on data type
        cmap = "viridis"
        if "dem" in key or "slope" in key:
            cmap = "terrain"
        elif "acc" in key:
            cmap = "Blues"
        elif "dir" in key:
            cmap = "Spectral"
        elif "twi" in key:
            cmap = "YlGnBu"

        img = ax.imshow(plot_arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(img, ax=ax, shrink=0.7)

        out_path = os.path.join(out_dir, f"{key}.png")
        print(f"Saving visualization to: {out_path}")
        fig.savefig(out_path, bbox_inches="tight")
        
        # Only show if running interactively
        try:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close(fig)
        except Exception: 
            plt.close(fig)

def visualize_vegetation_data(out_dir: str = "outputs") -> None:
    """Copy and display vegetation visualizations from data_vegetation directory."""
    os.makedirs(out_dir, exist_ok=True)
    
    # FIXED: Use absolute path or ensure current working directory is correct
    # If this path is relative, it depends on where you run python from.
    veg_data_dir = Path(os.getcwd()) / "fp" / "modules" / "vegetation" / "data_vegetation"
    
    # Fallback if running from a different root
    if not veg_data_dir.exists():
         veg_data_dir = Path("data_vegetation")

    rgb_source = veg_data_dir / "rgb_image.png"
    ndvi_source = veg_data_dir / "ndvi_image.png"
    
    # Copy files to output directory
    if rgb_source.exists():
        shutil.copy(rgb_source, os.path.join(out_dir, "rgb_image.png"))
    else:
        print(f"DEBUG: Could not find {rgb_source}")
    
    if ndvi_source.exists():
        shutil.copy(ndvi_source, os.path.join(out_dir, "ndvi_image.png"))
    
    # Display both images side by side if available
    if rgb_source.exists() and ndvi_source.exists():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # RGB image
        rgb_img = plt.imread(rgb_source)
        ax1.imshow(rgb_img)
        ax1.set_title('Sentinel-2 RGB Composite (L2A)', fontsize=14)
        ax1.axis('off')
        
        # NDVI image
        if ndvi_source.exists():
            ndvi_img = plt.imread(ndvi_source)
            ax2.imshow(ndvi_img)
            ax2.set_title('NDVI (Normalized Difference Vegetation Index)', fontsize=14)
            ax2.axis('off')
        
        plt.tight_layout()
        combined_path = os.path.join(out_dir, "vegetation_combined.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        
        try:
            plt.show(block=False)
            plt.pause(1)
            plt.close(fig)
        except Exception:
            plt.close(fig)
        
        print(f"Vegetation visualizations saved to {out_dir}")
    else:
        print(f"Warning: Vegetation images not found in {veg_data_dir}")

def visualize_and_save_all(integrated: Dict[str, Any], prediction: Dict[str, Any], out_dir: str = "outputs"):
    # Visualize topography data
    if "topography" in integrated and isinstance(integrated["topography"], dict):
        print("Visualizing Topography...")
        visualize_topography_data_separate(integrated["topography"], out_dir=out_dir)
    
    # Visualize vegetation data
    print("Visualizing Vegetation...")
    visualize_vegetation_data(out_dir=out_dir)
