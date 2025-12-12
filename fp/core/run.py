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
    """Ensures output directory exists."""
    os.makedirs(out_dir, exist_ok=True)

def run(coords: Dict[str, float], time: Optional[str], out_dir: str = "outputs") -> Dict[str, Any]:
    """
    Main pipeline: Acquires data, integrates spatially, predicts flood risk.
    
    Args:
        coords: Dictionary with 'lat', 'lon' keys
        time: ISO timestamp (optional)
        out_dir: Output directory for results
    
    Returns:
        Dictionary containing all module outputs
    """
    # Ensure output directory exists and is absolute
    out_dir = os.path.abspath(out_dir)
    _ensure_out_dir(out_dir)
    
    print("\n" + "="*70)
    print("FLOOD PREDICTION SYSTEM - PIPELINE START")
    print("="*70)
    
    print("\n--- Phase 1: Data Acquisition & Processing ---")
    
    # 1. Topography: Returns dict with DEM arrays AND 'profile' metadata
    print("\n[1/3] Fetching Topography Data...")
    topo = topographic_module(out_dir=out_dir, visualize=False)
    
    if topo.get("dem_filled") is not None:
        print(f"✓ Topography loaded: {topo['dem_filled'].shape}")
    else:
        print("✗ Topography loading failed!")
    
    # 2. Vegetation: Returns dict with NDVI/RGB arrays AND 'profile' metadata
    print("\n[2/3] Fetching Vegetation Data...")
    veg = get_vegetation_map(coords=coords, time=time or "", is_single_run=True)
    
    if veg.get("ndvi") is not None:
        print(f"✓ Vegetation loaded: {veg['ndvi'].shape}")
    else:
        print("✗ Vegetation loading failed!")
    
    # 3. Meteorology: Returns dict with precip/soil_moisture arrays AND 'profile'
    print("\n[3/3] Fetching Meteorological Data...")
    met = meteorological_module(coords=coords, time=time or "")
    
    if met.get("precip") is not None:
        precip_stats = {
            'min': met["precip"].min(),
            'max': met["precip"].max(),
            'mean': met["precip"].mean()
        }
        print(f"✓ Meteorology loaded: {met['precip'].shape}")
        print(f"  Precipitation stats: min={precip_stats['min']:.2f}, "
              f"max={precip_stats['max']:.2f}, mean={precip_stats['mean']:.2f} mm")
    else:
        print("⚠ Meteorology data unavailable - will use defaults")
    
    print("\n" + "="*70)
    print("--- Phase 2: Data Integration & Spatial Alignment ---")
    print("="*70)
    
    # 4. Integrate: Spatially aligns all layers to a common 30m grid
    integrated = integrate_data(
        topography=topo, 
        vegetation=veg, 
        meteorology=met,
        out_dir=out_dir,
        target_resolution=30.0
    )
    
    if integrated.get("met_precip") is not None:
        print(f"✓ Integrated precipitation: {integrated['met_precip'].shape}")
        print(f"  Mean: {integrated['met_precip'].mean():.2f} mm")
    else:
        print("⚠ Precipitation not integrated (using defaults)")
    
    print("\n" + "="*70)
    print("--- Phase 3: Flood Prediction & Visualization ---")
    print("="*70)
    
    # 5. Predict with hourly simulation
    prediction = predict_flood_risk(integrated, out_dir=out_dir)
    
    if prediction.get("probability_map") is not None:
        prob_stats = {
            'min': prediction["probability_map"].min(),
            'max': prediction["probability_map"].max(),
            'mean': prediction["probability_map"].mean()
        }
        print(f"\n✓ Flood prediction complete")
        print(f"  Risk probability: min={prob_stats['min']:.3f}, "
              f"max={prob_stats['max']:.3f}, mean={prob_stats['mean']:.3f}")
        print(f"  Hourly snapshots: {len(prediction.get('hourly_maps', []))} frames")
    
    # 6. Visualize all results
    print("\n--- Phase 4: Generating Visualizations ---")
    visualize_and_save_all(integrated, prediction, out_dir=out_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {out_dir}")
    print(f"  - Integrated data: integrated_comparison_3panel.png")
    print(f"  - Flood prediction: flood_probability.png")
    print(f"  - Hourly simulation: hourly_simulation/hourly_progression.png")
    print(f"  - Individual frames: hourly_simulation/hour_XX.png (24 files)")
    
    return {
        "topography": topo, 
        "vegetation": veg, 
        "meteorology": met, 
        "integrated": integrated, 
        "prediction": prediction
    }

def run_time_based(interval: int, coords: Dict[str, float], out_dir: str = "outputs") -> None:
    """
    Runs the pipeline repeatedly at specified intervals.
    
    Args:
        interval: Time between runs in seconds
        coords: Location coordinates
        out_dir: Output directory
    """
    _ensure_out_dir(out_dir)
    
    print(f"\n{'='*70}")
    print(f"TIME-BASED SCHEDULER STARTED")
    print(f"{'='*70}")
    print(f"Interval: {interval} seconds ({interval/3600:.1f} hours)")
    print(f"Location: {coords}")
    print(f"Output: {out_dir}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*70}\n")
    
    run_count = 0
    
    while True:
        try:
            run_count += 1
            print(f"\n[Run #{run_count}] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = run(coords=coords, time=None, out_dir=out_dir)
            
            print(f"\n[Run #{run_count}] Complete. Sleeping for {interval}s...")
            print(f"Next run at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + interval))}")
            
        except KeyboardInterrupt:
            print("\n\nScheduler stopped by user.")
            break
        except Exception as e:
            print(f"\n✗ [Run #{run_count}] Error during pipeline: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nRetrying in {interval}s...")
        
        time.sleep(interval)

if __name__ == "__main__":
    # Example single run
    test_coords = {
        'lat': 50.433,
        'lon': 16.653
    }
    
    run(coords=test_coords, time=None, out_dir="outputs")