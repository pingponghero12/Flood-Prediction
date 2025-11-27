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
    _ensure_out_dir(out_dir)

    topo = topographic_module(out_dir=out_dir, visualize=False)
    veg = get_vegetation_map(coords=coords, time=time or "", is_single_run=True)
    met = meteorological_module(coords=coords, time=time or "")

    integrated = integrate_data(topography=topo, vegetation=veg, meteorology=met, out_dir=out_dir)

    prediction = predict_flood_risk(integrated)
    visualize_and_save_all(integrated, prediction, out_dir=out_dir)

    return {"topography": topo, "vegetation": veg, "meteorology": met, "integrated": integrated, "prediction": prediction}

def run_time_based(interval: int, coords: Dict[str, float], out_dir: str = "outputs") -> None:
    _ensure_out_dir(out_dir)
    while True:
        run(coords=coords, time=None, out_dir=out_dir)
        time.sleep(interval)

