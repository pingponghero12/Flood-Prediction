from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def predict_flood_risk(integrated_data: Dict[str, Any], out_dir: str = "outputs") -> Dict[str, Any]:
    """
    Improved flood prediction with hourly simulation and visualization.
    Considers terrain slope, vegetation, and actual precipitation data.
    """
    print("\n=== Flood Prediction Module (Hourly Simulation) ===")
    
    os.makedirs(out_dir, exist_ok=True)
    hourly_dir = Path(out_dir) / "hourly_simulation"
    hourly_dir.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------------------
    # 1. Load DEM and compute slope
    # ---------------------------------------------------------------------
    dem = integrated_data.get("topo_dem_filled")
    slope = integrated_data.get("topo_slope")
    
    if dem is None:
        print("[Prediction] ERROR: Missing DEM — cannot compute flood risk.")
        return {"probability_map": None, "hourly_maps": []}
    
    h, w = dem.shape
    print(f"[Prediction] DEM loaded: {h} x {w}")
    
    # Normalize DEM (lower = more susceptible to flooding)
    dem_min = np.nanmin(dem)
    dem_max = np.nanmax(dem)
    dem_norm = (dem - dem_min) / (dem_max - dem_min + 1e-6)
    
    # Lower terrain = higher risk
    elevation_risk = 1.0 - dem_norm
    
    # ---------------------------------------------------------------------
    # 2. Slope factor (steeper = faster runoff = less accumulation)
    # ---------------------------------------------------------------------
    if slope is not None:
        # Normalize slope to 0-1 (assuming max reasonable slope ~45 degrees)
        slope_norm = np.clip(slope / 45.0, 0, 1)
        # Higher slope = more runoff = lower accumulation locally
        slope_factor = slope_norm
        print("[Prediction] Slope factor computed.")
    else:
        slope_factor = np.full((h, w), 0.3)
        print("[Prediction] Slope missing → using default slope factor.")
    
    # ---------------------------------------------------------------------
    # 3. Vegetation (NDVI) - acts as infiltration buffer
    # ---------------------------------------------------------------------
    ndvi = integrated_data.get("veg_ndvi")
    if ndvi is not None:
        # Higher NDVI = better infiltration = lower surface accumulation
        ndvi_norm = np.clip(ndvi, 0, 1)
        infiltration_factor = ndvi_norm
        print("[Prediction] Vegetation (NDVI) included.")
    else:
        infiltration_factor = np.full((h, w), 0.3)
        print("[Prediction] NDVI missing → default infiltration factor used.")
    
    # ---------------------------------------------------------------------
    # 4. Precipitation data
    # ---------------------------------------------------------------------
    precip = integrated_data.get("met_precip")
    
    if precip is not None:
        precip_max = np.nanmax(precip)
        if precip_max > 0:
            print(f"[Prediction] Precipitation data found. Max: {precip_max:.2f} mm/24h")
            # Convert 24h total to hourly rate (simple linear assumption)
            hourly_rain = precip / 24.0
        else:
            print("[Prediction] WARNING: Precipitation data is all zeros!")
            hourly_rain = np.full((h, w), 0.5)  # Default light rain for testing
    else:
        print("[Prediction] No precipitation data → using default 10mm/24h")
        hourly_rain = np.full((h, w), 10.0 / 24.0)
    
    # ---------------------------------------------------------------------
    # 5. 24-hour accumulation simulation with hourly snapshots
    # ---------------------------------------------------------------------
    print("[Prediction] Starting 24h hourly accumulation simulation...")
    
    water_accumulated = np.zeros((h, w), dtype=float)
    hourly_maps = []
    
    # Runoff coefficient: combines slope and infiltration
    # More slope = more runoff, more vegetation = less runoff
    runoff_coeff = (slope_factor * 0.7 + (1 - infiltration_factor) * 0.3)
    runoff_coeff = np.clip(runoff_coeff, 0, 1)
    
    # Infiltration rate (how much water soaks into ground per hour)
    infiltration_rate = infiltration_factor * 0.15  # up to 15% per hour for dense veg
    
    for hour in range(24):
        # Add rainfall for this hour
        water_accumulated += hourly_rain
        
        # Remove water due to runoff (flows away based on slope)
        runoff_loss = water_accumulated * runoff_coeff * 0.25
        water_accumulated -= runoff_loss
        
        # Remove water due to infiltration (soaks into ground)
        infiltration_loss = water_accumulated * infiltration_rate
        water_accumulated -= infiltration_loss
        
        # Ensure no negative values
        water_accumulated = np.clip(water_accumulated, 0, None)
        
        # Store snapshot
        hourly_maps.append(water_accumulated.copy())
        
        if hour % 6 == 0:
            print(f"[Prediction] Hour {hour:02d} — mean water level: {water_accumulated.mean():.4f} mm")
    
    print("[Prediction] Accumulation simulation complete.")
    
    # ---------------------------------------------------------------------
    # 6. Final flood probability map (normalized)
    # ---------------------------------------------------------------------
    max_water = np.nanmax(water_accumulated)
    if max_water > 0:
        water_norm = water_accumulated / max_water
    else:
        water_norm = water_accumulated
    
    # Combine factors: elevation risk + water accumulation
    prob = (
        0.4 * elevation_risk +
        0.6 * water_norm
    )
    prob = np.clip(prob, 0, 1)
    
    print("[Prediction] Final probability map computed.")
    
    # ---------------------------------------------------------------------
    # 7. Visualize hourly progression (every 3 hours + final)
    # ---------------------------------------------------------------------
    visualize_hourly_progression(hourly_maps, hourly_dir)
    
    print("=== Prediction Complete ===\n")
    
    return {
        "probability_map": prob,
        "hourly_maps": hourly_maps,
        "final_water_level": water_accumulated
    }


def visualize_hourly_progression(hourly_maps: List[np.ndarray], out_dir: Path):
    """
    Creates visualizations showing water accumulation every 3 hours.
    """
    print("[Visualization] Generating hourly simulation snapshots...")
    
    hours_to_plot = [0, 3, 6, 9, 12, 15, 18, 21, 23]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    vmax = max(np.nanmax(m) for m in hourly_maps)
    
    for idx, hour in enumerate(hours_to_plot):
        ax = axes[idx]
        water_map = hourly_maps[hour]
        
        im = ax.imshow(water_map, cmap='Blues', vmin=0, vmax=vmax)
        ax.set_title(f"Hour {hour:02d}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Water (mm)')
    
    plt.suptitle('24-Hour Water Accumulation Simulation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = out_dir / "hourly_progression.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved: {save_path}")
    
    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
    except:
        plt.close(fig)
    
    # Also save individual frames for animation potential
    for hour in range(24):
        fig_single, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(hourly_maps[hour], cmap='Blues', vmin=0, vmax=vmax)
        ax.set_title(f"Hour {hour:02d} - Water Accumulation", fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Water Level (mm)')
        
        frame_path = out_dir / f"hour_{hour:02d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig_single)
    
    print(f"[Visualization] Saved 24 individual hourly frames in {out_dir}")