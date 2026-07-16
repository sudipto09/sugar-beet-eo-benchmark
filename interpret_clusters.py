import numpy as np
import pickle
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATA_DIR  = "/home/sudiptochakraborty/praktikum_cv/Data"

# Load everything
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    results = pickle.load(f)

df = pd.read_csv(f"{DATA_DIR}/benchmark_results_full.csv")
gdf = gpd.read_file(f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson")
gdf = gdf.to_crs("EPSG:32632")

# Merge
gdf["prithvi_cluster"] = df["prithvi_cluster"].values
gdf["ndvi_cluster"]    = df["ndvi_cluster"].values
gdf["ndvi_apr"]        = df["ndvi_apr"].values
gdf["ndvi_jun"]        = df["ndvi_jun"].values
gdf["ndvi_aug"]        = df["ndvi_aug"].values
gdf["ndvi_change"]     = df["ndvi_aug"].values - df["ndvi_jun"].values

# ============================================================
# Q: What does each Prithvi cluster look like in terms of
#    field area, NDVI level, and spatial location?
# ============================================================
print("Prithvi cluster characteristics:")
print("="*70)
for c in range(5):
    mask = gdf["prithvi_cluster"] == c
    sub = gdf[mask]
    print(f"\nCluster {c} (n={mask.sum()}):")
    print(f"  Mean area:      {sub['area_ha'].mean():.2f} ha")
    print(f"  Mean NDVI Apr:  {sub['ndvi_apr'].mean():.3f}")
    print(f"  Mean NDVI Jun:  {sub['ndvi_jun'].mean():.3f}")
    print(f"  Mean NDVI Aug:  {sub['ndvi_aug'].mean():.3f}")
    print(f"  NDVI change Jun→Aug: {sub['ndvi_change'].mean():.3f}")

print("\n\nNDVI cluster characteristics:")
print("="*70)
for c in range(5):
    mask = gdf["ndvi_cluster"] == c
    sub = gdf[mask]
    print(f"\nCluster {c} (n={mask.sum()}):")
    print(f"  Mean area:      {sub['area_ha'].mean():.2f} ha")
    print(f"  Mean NDVI Apr:  {sub['ndvi_apr'].mean():.3f}")
    print(f"  Mean NDVI Jun:  {sub['ndvi_jun'].mean():.3f}")
    print(f"  Mean NDVI Aug:  {sub['ndvi_aug'].mean():.3f}")
    print(f"  NDVI change Jun→Aug: {sub['ndvi_change'].mean():.3f}")

# ============================================================
# SPATIAL MAP — where are Prithvi's clusters geographically?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

# Prithvi spatial map
gdf_wgs = gdf.to_crs("EPSG:4326")
for c in range(5):
    mask = gdf_wgs["prithvi_cluster"] == c
    sub = gdf_wgs[mask]
    sub.plot(ax=axes[0], color=colors[c], alpha=0.7,
             label=f"Cluster {c} (n={mask.sum()})")
axes[0].set_title("Prithvi EO 2.0 — Spatial Cluster Distribution",
                  fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

# NDVI spatial map
for c in range(5):
    mask = gdf_wgs["ndvi_cluster"] == c
    sub = gdf_wgs[mask]
    sub.plot(ax=axes[1], color=colors[c], alpha=0.7,
             label=f"Cluster {c} (n={mask.sum()})")
axes[1].set_title("NDVI Baseline — Spatial Cluster Distribution",
                  fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xlabel("Longitude")

plt.suptitle("Spatial Distribution of Clusters\nSugar Beet Fields, Franconia 2022",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/spatial_cluster_map.png", dpi=120, bbox_inches='tight')
print(f"\nSpatial map saved: Data/spatial_cluster_map.png")

# ============================================================
# KEY QUESTION: Do Prithvi clusters correlate with field size?
# ============================================================
print("\n\nPrithvi cluster vs field area (testing if Prithvi clusters by size):")
for c in range(5):
    mask = gdf["prithvi_cluster"] == c
    print(f"  Cluster {c}: mean={gdf[mask]['area_ha'].mean():.2f} ha, "
          f"std={gdf[mask]['area_ha'].std():.2f}")
