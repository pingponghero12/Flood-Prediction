import rasterio
import numpy as np
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from whitebox import WhiteboxTools
import os

# ───────────────────────────────────────────────────────────
# KONFIGURACJA
# ───────────────────────────────────────────────────────────
# Musiałem dać absolute path bo whitebox odpala się gdzieś indziej zmieńcie sobie
DEM_PATH = os.path.abspath("data.tif") 
OUT_DIR = os.path.abspath("outputs")

os.makedirs(OUT_DIR, exist_ok=True)

DEM_FILLED = f"{OUT_DIR}/dem_filled.tif"
FLOW_DIR = f"{OUT_DIR}/flow_dir.tif"
FLOW_ACC = f"{OUT_DIR}/flow_acc.tif"
SLOPE = f"{OUT_DIR}/slope.tif"
TWI = f"{OUT_DIR}/twi.tif"

# ───────────────────────────────────────────────────────────
# INICJALIZACJA WHITEBOXA
# ───────────────────────────────────────────────────────────
wbt = WhiteboxTools()
wbt.work_dir = OUT_DIR

# ───────────────────────────────────────────────────────────
# 1. Wczytanie DEM
# ───────────────────────────────────────────────────────────
print("Wczytywanie DEM...")
try:
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1)
        profile = src.profile
except rasterio.RasterioIOError as e:
    print(f"BŁĄD: Nie można wczytać pliku DEM '{DEM_PATH}'. Upewnij się, że plik istnieje w katalogu skryptu. Szczegóły: {e}")
    # Kończymy działanie skryptu, jeśli plik DEM nie istnieje
    exit()


# ───────────────────────────────────────────────────────────
# 2. Wypełnianie depresji (Fill Depressions)
# ───────────────────────────────────────────────────────────
print("Wypełnianie depresji...")
wbt.fill_depressions(
    dem=DEM_PATH, 
    output=DEM_FILLED,
)

# ───────────────────────────────────────────────────────────
# 3. Kierunki przepływu (D8 Flow Direction)
# ───────────────────────────────────────────────────────────
print("Obliczanie kierunków przepływu (D8)...")
wbt.d8_pointer(
    dem=DEM_FILLED,
    output=FLOW_DIR
)

# ───────────────────────────────────────────────────────────
# 4. Akumulacja przepływu (Flow Accumulation D8)
# ───────────────────────────────────────────────────────────
print("Obliczanie akumulacji przepływu...")
wbt.d8_flow_accumulation(
    i=DEM_FILLED,
    output=FLOW_ACC,
    out_type='cells'   # liczba komórek dopływających
)

# ───────────────────────────────────────────────────────────
# 5. Spadek terenu (Slope)
# ───────────────────────────────────────────────────────────
print("Obliczanie spadku...")
wbt.slope(
    dem=DEM_FILLED,
    output=SLOPE,
    zfactor=None   
)

# ───────────────────────────────────────────────────────────
# 6. Obliczenie TWI
# TWI = ln( As / tan(slope) )
# As = specific catchment area = flow_acc * cellsize
# slope w radianach
# ───────────────────────────────────────────────────────────

print("Obliczanie TWI...")

with rasterio.open(FLOW_ACC) as acc_src:
    acc = acc_src.read(1)
    cellsize = acc_src.res[0]

with rasterio.open(SLOPE) as sl_src:
    slope_deg = sl_src.read(1)

slope_rad = np.radians(slope_deg)

# unikamy dzielenia przez zero
slope_rad[slope_rad == 0] = np.nan

# Specific Catchment Area
sca = acc * cellsize

# TWI
twi = np.log(sca / np.tan(slope_rad))

# zapisujemy TWI jako GeoTIFF
twi_profile = profile.copy()
twi_profile.update(dtype=rasterio.float32, count=1)

with rasterio.open(TWI, "w", **twi_profile) as dst:
    dst.write(twi.astype(np.float32), 1)

print("Analiza zakończona. Pliki zapisane w:", OUT_DIR)
