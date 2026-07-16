
import rasterio
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATES = ["20220411", "20220514", "20220623", 
         "20220725", "20220812", "20221115"]

DATE_LABELS = ["Apr 11", "May 14", "Jun 23", 
               "Jul 25", "Aug 12", "Nov 15"]

BASE_DIR = Path("/home/sudiptochakraborty/praktikum_cv/Data/sentinel2/franconia")

ndvi_values = []
ndre_values = []

print("Computing spectral indices from real Sentinel-2 data...\n")

for date in DATES:
    scene_dir = BASE_DIR / date
    
    # Load B04 (Red, 10m)
    with rasterio.open(scene_dir / "B04.tif") as src:
        B04 = src.read(1).astype(np.float32)
        target_shape = (src.height, src.width)
    
    # Load B08 (NIR, 10m)
    with rasterio.open(scene_dir / "B08.tif") as src:
        B08 = src.read(1).astype(np.float32)
    
    # Load B05 (Red Edge, 20m) — resample to 10m
    with rasterio.open(scene_dir / "B05.tif") as src:
        B05 = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear
        ).astype(np.float32)
    
    # Scale reflectance
    B04 = B04 / 10000.0
    B08 = B08 / 10000.0
    B05 = B05 / 10000.0
    
    eps = 1e-8
    
    # NDVI
    ndvi = (B08 - B04) / (B08 + B04 + eps)
    ndvi = np.clip(ndvi, -1, 1)
    
    # NDRE
    ndre = (B08 - B05) / (B08 + B05 + eps)
    ndre = np.clip(ndre, -1, 1)
    
    # Remove invalid pixels
    valid = (B04 > 0) & (B08 > 0) & (ndvi > 0)
    
    ndvi_mean = ndvi[valid].mean()
    ndre_mean = ndre[valid].mean()
    
    ndvi_values.append(ndvi_mean)
    ndre_values.append(ndre_mean)
    
    print(f"{date}: NDVI={ndvi_mean:.3f}  NDRE={ndre_mean:.3f}")

# Save results
results_df = pd.DataFrame({
    "date": DATES,
    "date_label": DATE_LABELS,
    "ndvi": ndvi_values,
    "ndre": ndre_values
})
results_df.to_csv("/home/sudiptochakraborty/praktikum_cv/Data/real_ndvi_results.csv", 
                  index=False)
print("\nSaved to /home/sudiptochakraborty/praktikum_cv/Data/real_ndvi_results.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(DATE_LABELS, ndvi_values,
             color='#1B5E20', linewidth=2.5,
             marker='o', markersize=8)
axes[0].fill_between(range(len(DATE_LABELS)), ndvi_values,
                     alpha=0.15, color='#1B5E20')
axes[0].set_title('Real NDVI — Franconia 2022 (Sentinel-2)',
                  fontweight='bold', fontsize=13)
axes[0].set_ylabel('Mean NDVI')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=30)

axes[1].plot(DATE_LABELS, ndre_values,
             color='#4A0E8F', linewidth=2.5,
             marker='s', markersize=8)
axes[1].fill_between(range(len(DATE_LABELS)), ndre_values,
                     alpha=0.15, color='#4A0E8F')
axes[1].set_title('Real NDRE — Franconia 2022 (Sentinel-2)',
                  fontweight='bold', fontsize=13)
axes[1].set_ylabel('Mean NDRE')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('/home/sudiptochakraborty/praktikum_cv/Data/real_ndvi_ndre_2022.png',
            dpi=120, bbox_inches='tight')
print("Plot saved to data/real_ndvi_ndre_2022.png")
