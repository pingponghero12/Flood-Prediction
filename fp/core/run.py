from typing import Dict, Any, Optional
import os
import time
from fp.modules.topographic.topographic_module import topographic_module
from fp.modules.vegetation.vegetation_module import get_vegetation_map
from fp.modules.meteorological.meteorological_module import meteorological_module
from fp.modules.data_integration.data_integration_module import integrate_data
from fp.modules.data_visualization.data_visualization_module import visualize_and_save_all
from fp.prediction.prediction_algorithm import predict_flood_risk

def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

def run(coords: Dict[str, float], time: Optional[str], out_dir: str = "outputs") -> Dict[str, Any]:
    # Ensure output directory exists and is absolute
    out_dir = os.path.abspath(out_dir)
    _ensure_out_dir(out_dir)

    print("\n--- Phase 1: Data Acquisition & Processing ---")
    
    # 1. Topography: Returns dict with DEM arrays AND 'profile' metadata
    print("Fetching Topography Data...")
    topo = topographic_module(out_dir=out_dir, visualize=False)
    
    # 2. Vegetation: Returns dict with NDVI/RGB arrays AND 'profile' metadata
    print("Fetching Vegetation Data...")
    veg = get_vegetation_map(coords=coords, time=time or "", is_single_run=True)
    
    # 3. Meteorology: Returns dict with precip/temp arrays
    print("Fetching Meteorological Data...")
    met = meteorological_module(coords=coords, time=time or "")

    print("\n--- Phase 2: Data Integration & Alignment ---")
    
    # 4. Integrate: Spatially aligns all layers to a common 30m grid
    #    (Matches the new signature we defined)
    integrated = integrate_data(
        topography=topo, 
        vegetation=veg, 
        meteorology=met,  # Pass this through if you updated integration to handle it, otherwise it's just stored
        out_dir=out_dir,
        target_resolution=30.0
    )

    print("\n--- Phase 3: Prediction & Visualization ---")
    
    # 5. Predict
    prediction = predict_flood_risk(integrated)
    
    # 6. Visualize
    visualize_and_save_all(integrated, prediction, out_dir=out_dir)

    return {
        "topography": topo, 
        "vegetation": veg, 
        "meteorology": met, 
        "integrated": integrated, 
        "prediction": prediction
    }

def run_time_based(interval: int, coords: Dict[str, float], out_dir: str = "outputs") -> None:
    _ensure_out_dir(out_dir)
    print(f"Starting time-based run. Interval: {interval} seconds.")
    while True:
        try:
            print(f"\n[Scheduler] Running pipeline at {time.ctime()}")
            run(coords=coords, time=None, out_dir=out_dir)
            print(f"[Scheduler] Run complete. Sleeping for {interval}s...")
        except Exception as e:
            print(f"[Scheduler] Error during run: {e}")
            
        time.sleep(interval)
