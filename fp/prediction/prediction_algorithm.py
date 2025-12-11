from typing import Dict, Any
import numpy as np

def predict_flood_risk(integrated_data: Dict[str, Any]) -> Dict[str, Any]:
    print("\n=== Flood Prediction Module ===")

    # ---------------------------------------------------------------------
    # 1. Pobranie DEM
    # ---------------------------------------------------------------------
    dem = integrated_data.get("topo_dem_filled")
    if dem is None:
        print("[Prediction] ERROR: Missing DEM — cannot compute flood risk.")
        return {"probability_map": None}

    h, w = dem.shape
    print(f"[Prediction] DEM loaded: {h} x {w}")

    # Normalizacja DEM (wyższe = suchsze)
    dem_min = np.nanmin(dem)
    dem_max = np.nanmax(dem)
    dem_norm = (dem - dem_min) / (dem_max - dem_min + 1e-6)

    # Niższy teren = większe ryzyko → odwracamy skalę
    terrain_factor = 1.0 - dem_norm
    print("[Prediction] Terrain factor computed.")

    # ---------------------------------------------------------------------
    # 2. Vegetacja (NDVI)
    # ---------------------------------------------------------------------
    ndvi = integrated_data.get("veg_ndvi")
    if ndvi is not None:
        # im wyższy NDVI tym łatwiej infiltruje, więc niższe ryzyko
        vegetation_factor = 1.0 - np.clip(ndvi, 0, 1)
        print("[Prediction] Vegetation (NDVI) included.")
    else:
        vegetation_factor = np.full((h, w), 0.5)
        print("[Prediction] NDVI missing → default vegetation factor used.")

    # ---------------------------------------------------------------------
    # 3. Meteorologia — opad
    # ---------------------------------------------------------------------
    met = integrated_data.get("meteorology", {})
    precip = met.get("precipitation") if isinstance(met, dict) else None

    if precip is not None:
        pmax = np.nanmax(precip)
        precip_norm = precip / (pmax + 1e-6)
        print("[Prediction] Precipitation layer included.")
    else:
        precip_norm = np.full((h, w), 0.3)
        print("[Prediction] No precipitation found → default used.")

    # ---------------------------------------------------------------------
    # 4. Mini-symulacja akumulacji wody (24h)
    # ---------------------------------------------------------------------
    print("[Prediction] Starting 24h accumulation mini-simulation...")

    water = np.zeros((h, w), dtype=float)
    runoff_coeff = (terrain_factor * 0.6 + vegetation_factor * 0.4)

    # Hourly simulation
    for hour in range(24):
        # Dodaj deszcz
        water += precip_norm * 0.05  # każda godzina dodaje trochę podniesienia poziomu

        # Spływ (im większy runoff, tym więcej odpływa)
        water -= runoff_coeff * 0.03

        # Brak wartości ujemnych
        water = np.clip(water, 0, None)

        if hour % 6 == 0:
            print(f"[Prediction] Hour {hour:02d} — mean water={water.mean():.4f}")

    print("[Prediction] Accumulation simulation complete.")

    # ---------------------------------------------------------------------
    # 5. Końcowa mapa prawdopodobieństwa
    # ---------------------------------------------------------------------
    # Kombinacja czynników
    prob = (
        0.5 * terrain_factor +
        0.2 * vegetation_factor +
        0.3 * np.clip(water / (water.max() + 1e-6), 0, 1)
    )

    prob = np.clip(prob, 0, 1)
    print("[Prediction] Final probability map computed.")
    print("=== Prediction Complete ===\n")

    return {"probability_map": prob}
