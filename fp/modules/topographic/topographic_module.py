from typing import Dict, Any
import os
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from whitebox import WhiteboxTools

from fp.modules.data_visualization.data_visualization_module import (
    visualize_topography_data_separate,
)

DEFAULT_OUT_DIR = "outputs"
DEFAULT_DEM = "data.tif"

DEFAULT_FILES = {
    "dem_filled": "dem_filled.tif",
    "flow_dir": "flow_dir.tif",
    "flow_acc": "flow_acc.tif",
    "slope": "slope.tif",
    "twi": "twi.tif",
}

def _read_raster_maybe(path: str):
    """Reads a raster if it exists, else returns None."""
    try:
        with rasterio.open(path) as src:
            return src.read(1)
    except RasterioIOError:
        return None
    except Exception:
        return None


def _run_whitebox(dem_path: str, out_dir: str):
    """Runs the WhiteboxTools workflow."""
    wbt = WhiteboxTools()
    wbt.work_dir = out_dir

    dem_filled = os.path.join(out_dir, DEFAULT_FILES["dem_filled"])
    flow_dir = os.path.join(out_dir, DEFAULT_FILES["flow_dir"])
    flow_acc = os.path.join(out_dir, DEFAULT_FILES["flow_acc"])
    slope = os.path.join(out_dir, DEFAULT_FILES["slope"])
    twi = os.path.join(out_dir, DEFAULT_FILES["twi"])

    # 1. Fill depressions
    wbt.fill_depressions(
        dem=dem_path,
        output=dem_filled,
    )

    # 2. Flow direction (D8)
    wbt.d8_pointer(
        dem=dem_filled,
        output=flow_dir
    )

    # 3. Flow accumulation (D8)
    wbt.d8_flow_accumulation(
        i=dem_filled,
        output=flow_acc,
        out_type="cells",
    )

    # 4. Slope
    wbt.slope(
        dem=dem_filled,
        output=slope,
        zfactor=None,
    )

    # 5. TWI calculation
    try:
        with rasterio.open(flow_acc) as acc_src:
            acc = acc_src.read(1)
            cellsize = acc_src.res[0]

        with rasterio.open(slope) as sl_src:
            slope_deg = sl_src.read(1)

        slope_rad = np.radians(slope_deg)
        slope_rad[slope_rad == 0] = np.nan

        sca = acc * cellsize
        twi_arr = np.log(sca / np.tan(slope_rad))

        # Save TWI
        profile = acc_src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(twi, "w", **profile) as dst:
            dst.write(twi_arr.astype(np.float32), 1)

    except rasterio.RasterioIOError:
        # If something goes wrong, create NaN placeholder
        twi_arr = np.full((256, 256), np.nan)
        with rasterio.open(twi, "w", driver="GTiff",
                           height=256, width=256,
                           count=1, dtype="float32") as dst:
            dst.write(twi_arr.astype(np.float32), 1)


def topographic_module(
    dem_path: str = DEFAULT_DEM,
    out_dir: str = DEFAULT_OUT_DIR,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Full topography workflow:
    - Runs WhiteboxTools analysis
    - Loads results from disk
    - Returns raw arrays
    - Optionally visualizes
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM file does not exist: {dem_path}")

    print(f"Running topographic analysis in: {out_dir}")
    _run_whitebox(dem_path, out_dir)

    # Load rasters
    data: Dict[str, Any] = {}
    for key, fname in DEFAULT_FILES.items():
        path = os.path.join(out_dir, fname)
        arr = _read_raster_maybe(path)
        if arr is None:
            arr = np.full((256, 256), np.nan)
        data[key] = {"array": arr, "path": path}

    if visualize:
        visualize_topography_data_separate(
            {k: v["array"] for k, v in data.items()},
            out_dir=out_dir
        )

    print("Topography analysis complete.")
    return {k: v["array"] for k, v in data.items()}


if __name__ == "__main__":
    topo = topographic_module(visualize=True)
    print("Topography maps:", list(topo.keys()))
