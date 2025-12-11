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

        # Convert to numeric array if possible
        try:
            arr = np.array(arr, dtype=float)
        except Exception:
            arr = None

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

def visualize_vegetation_data(integrated: Dict[str, Any], out_dir: str = "outputs") -> None:
    """Visualizes vegetation data (NDVI or RGB if available) directly from integrated arrays."""
    os.makedirs(out_dir, exist_ok=True)

    ndvi = integrated.get("veg_ndvi")
    rgb = integrated.get("veg_rgb")

    # ==========================
    # CASE 1: NDVI AVAILABLE
    # ==========================
    if ndvi is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            ndvi_arr = np.array(ndvi, dtype=float)
        except:
            ndvi_arr = None

        if ndvi_arr is None or np.all(np.isnan(ndvi_arr)):
            ax.text(0.5, 0.5, "NDVI Missing", ha="center", va="center",
                    fontsize=14, color="red")
            ax.axis("off")
        else:
            img = ax.imshow(ndvi_arr, cmap="YlGn")
            ax.set_title("NDVI (Vegetation Density)")
            ax.axis("off")
            plt.colorbar(img, ax=ax, shrink=0.7)

        out_path = os.path.join(out_dir, "ndvi_map.png")
        print(f"Saving NDVI visualization to: {out_path}")
        fig.savefig(out_path, bbox_inches="tight")

        try:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close(fig)
        except:
            plt.close(fig)

    # ==========================
    # CASE 2: RGB AVAILABLE
    # ==========================
    if rgb is not None:
        try:
            rgb_arr = np.array(rgb, dtype=np.uint8)
        except:
            rgb_arr = None

        if rgb_arr is not None and rgb_arr.ndim == 3:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(rgb_arr)
            ax.set_title("Vegetation RGB (Sentinel-2)")
            ax.axis("off")

            out_path = os.path.join(out_dir, "vegetation_rgb.png")
            print(f"Saving RGB visualization to: {out_path}")
            fig.savefig(out_path, bbox_inches="tight")

            try:
                plt.show(block=False)
                plt.pause(0.5)
                plt.close(fig)
            except:
                plt.close(fig)
        else:
            print("Warning: RGB data present, but invalid format")

    # ==========================
    # CASE 3: NOTHING AVAILABLE
    # ==========================
    if ndvi is None and rgb is None:
        print("Warning: No vegetation arrays in integrated data → skipping vegetation visualization.")

def visualize_prediction_data(prediction: Dict[str, Any], out_dir: str = "outputs") -> None:
    """Visualize flood probability map."""
    os.makedirs(out_dir, exist_ok=True)

    prob = prediction.get("probability_map")
    if prob is None:
        print("Warning: No probability map in prediction → skipping flood map.")
        return

    try:
        prob_arr = np.array(prob, dtype=float)
    except:
        print("Error: Probability map is not numeric.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(prob_arr, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_title("Flood Probability Map (0–1)")
    ax.axis("off")
    plt.colorbar(img, ax=ax, shrink=0.7)

    out_path = os.path.join(out_dir, "flood_probability.png")
    print(f"Saving flood probability map to: {out_path}")
    fig.savefig(out_path, bbox_inches="tight")

    try:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close(fig)
    except:
        plt.close(fig)


def visualize_and_save_all(integrated: Dict[str, Any], prediction: Dict[str, Any], out_dir: str = "outputs"):

    # 1) Topography
    if "topography" in integrated and isinstance(integrated["topography"], dict):
        print("Visualizing Topography...")
        visualize_topography_data_separate(integrated["topography"], out_dir=out_dir)

    # 2) Vegetation
    print("Visualizing Vegetation...")
    visualize_vegetation_data(integrated, out_dir=out_dir)

    # 3) Flood Prediction
    print("Visualizing Flood Prediction...")
    visualize_prediction_data(prediction, out_dir=out_dir)


