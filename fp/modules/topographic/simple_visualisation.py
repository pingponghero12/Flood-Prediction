import rasterio
import matplotlib.pyplot as plt
import os
import numpy as np

# ───────────────────────────────────────────────────────────
# KONFIGURACJA ŚCIEŻEK
# ───────────────────────────────────────────────────────────
OUT_DIR = os.path.abspath("outputs") # Katalog, w którym zapisane są wszystkie pliki
FILES_TO_VISUALIZE = {
    "1. Wypełniony DEM": f"{OUT_DIR}/dem_filled.tif",
    "2. Kierunek Przepływu (D8)": f"{OUT_DIR}/flow_dir.tif",
    "3. Akumulacja Przepływu": f"{OUT_DIR}/flow_acc.tif",
    "4. Spadek Terenu (Stopnie)": f"{OUT_DIR}/slope.tif",
    "5. Wskaźnik Zawilgocenia (TWI)": f"{OUT_DIR}/twi.tif",
}

# ───────────────────────────────────────────────────────────
# FUNKCJA WIZUALIZUJĄCA
# ───────────────────────────────────────────────────────────
def visualize_topography_data_separate():
    """Wczytuje i wizualizuje każdy plik GeoTIFF w osobnym oknie."""
    
    print("Rozpoczynanie wizualizacji...")

    for title, filepath in FILES_TO_VISUALIZE.items():
        
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                
                # Ustawienie kolorystyki (colormap)
                cmap = 'viridis' # Domyślna

                if "DEM" in title or "Spadek" in title:
                    cmap = 'terrain'
                elif "Akumulacja" in title:
                    # Skala logarytmiczna dla akumulacji przepływu
                    data = np.log(data, where=(data>0))
                    data[data == -np.inf] = np.nan 
                    cmap = 'Blues'
                elif "Kierunek" in title: # Kierunek przepływu
                    data = np.where(data > 0, np.log2(data) + 1, 0)
                    cmap = 'Spectral'
                elif "TWI" in title:
                    cmap = 'YlGnBu'

                # Tworzenie nowego okna wykresu dla każdego pliku
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Rysowanie rastra
                img = ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=14)
                ax.axis('off') # Ukrycie osi

                # Dodanie paska kolorów
                plt.colorbar(img, ax=ax, shrink=0.7)
                
                plt.show() # Wyświetlenie bieżącego okna

        except rasterio.RasterioIOError:
            print(f"Ostrzeżenie: Plik nie znaleziony lub uszkodzony: {filepath}")
            # Nadal wyświetlamy puste okno z komunikatem o błędzie
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_title(f"BRAK PLIKU:\n{title}", color='red')
            ax.axis('off')
            plt.show()
        except Exception as e:
             print(f"Wystąpił nieoczekiwany błąd podczas wczytywania {filepath}: {e}")
             fig, ax = plt.subplots(1, 1, figsize=(10, 8))
             ax.set_title(f"BŁĄD: {title}", color='red')
             ax.axis('off')
             plt.show()

if __name__ == "__main__":
    # Zmieniono wywołanie funkcji na nową, dla oddzielnych wykresów
    visualize_topography_data_separate()