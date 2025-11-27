from typing import Dict, Any

def get_s2_data(coords: Dict[str, float], time: str, is_single_run: bool) -> Any:
    return None

def process_s2_to_vegetation_map(raw_s2_data: Any) -> Any:
    return None

def get_vegetation_map(coords: Dict[str, float], time: str, is_single_run: bool) -> Any:
    raw = get_s2_data(coords=coords, time=time, is_single_run=is_single_run)
    if not raw:
        import numpy as np
        return {"ndvi": np.full((256, 256), float("nan"))}
    return {"ndvi": process_s2_to_vegetation_map(raw)}
