from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt

def _safe_maybe_log(arr):
    if arr is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.array(arr, dtype=float)
        out[out <= 0] = np.nan
        return np.log(out)

def visualize_topography_data_separate(data: Dict[str, Any], out_dir: str = "outputs") -> None:
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

        if arr is None:
            ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center", fontsize=14, color="red")
            ax.axis("off")
            out_path = os.path.join(out_dir, f"{key}_missing.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            continue

        plot_arr = arr.copy()
        if key == "flow_acc":
            plot_arr = _safe_maybe_log(plot_arr)

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
        fig.savefig(out_path, bbox_inches="tight")
        try:
            plt.show()
        except Exception:
            plt.close(fig)

def visualize_and_save_all(integrated: Dict[str, Any], prediction: Dict[str, Any], out_dir: str = "outputs"):
    # simple visualization: if integrated has topography keys, pass to topo visualizer
    topo_keys = {k: v for k, v in integrated.items() if isinstance(v, (list, tuple, type(None))) or hasattr(v, "shape")}
    # fallback: integrated may be nested
    if "topography" in integrated and isinstance(integrated["topography"], dict):
        visualize_topography_data_separate(integrated["topography"], out_dir=out_dir)
