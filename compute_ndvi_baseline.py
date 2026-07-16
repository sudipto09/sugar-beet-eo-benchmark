import numpy as np
import pickle
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR  = "/home/sudiptochakraborty/praktikum_cv/Data"
SCENE_DIR = f"{DATA_DIR}/sentinel2/wuerzburg_core"
DATES     = ["20220416", "20220618", "20220807"]
DATE_LABELS = ["Apr 16", "Jun 18", "Aug 7"]

# Load fields
gdf = gpd.read_file(f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson")
with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
    scene_crs = src.crs
gdf = gdf.to_crs(scene_crs)

# ============================================================
# COMPUTE REAL NDVI (B08 NIR 10m / B04 Red 10m)
# ============================================================
print("Computing NDVI/NDRE per field (B04, B08 at 10m)...")

ndvi_features = []
ndvi_trajectories = []

for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
    field_geom = row.geometry
    field_ndvi = []
    field_ndre = []

    for date in DATES:
        try:
            with rasterio.open(f"{SCENE_DIR}/{date}/B04.tif") as src:
                b04, _ = rio_mask(src, [field_geom], crop=True,
                                  nodata=0, all_touched=True)
                b04 = b04[0].astype(np.float32)

            with rasterio.open(f"{SCENE_DIR}/{date}/B08.tif") as src:
                b08, _ = rio_mask(src, [field_geom], crop=True,
                                  nodata=0, all_touched=True)
                b08 = b08[0].astype(np.float32)

            eps = 1e-8
            ndvi = (b08 - b04) / (b08 + b04 + eps)
            valid = (b04 > 100) & (b08 > 100)  # threshold avoids near-zero noise

            if valid.sum() > 0:
                field_ndvi.append(float(ndvi[valid].mean()))
            else:
                field_ndvi.append(0.0)

        except Exception:
            field_ndvi.append(0.0)

    ndvi_trajectories.append(field_ndvi)
    # Feature vector: NDVI at each timestep + temporal change
    if len(field_ndvi) == 3:
        delta1 = field_ndvi[1] - field_ndvi[0]
        delta2 = field_ndvi[2] - field_ndvi[1]
        ndvi_features.append(field_ndvi + [delta1, delta2])
    else:
        ndvi_features.append(field_ndvi + [0.0, 0.0])

ndvi_features = np.array(ndvi_features)
ndvi_traj     = np.array(ndvi_trajectories)

print(f"NDVI features shape: {ndvi_features.shape}")
print(f"NDVI value range: {ndvi_traj.min():.3f} to {ndvi_traj.max():.3f}")
print(f"Mean NDVI per timestep: {ndvi_traj.mean(axis=0).round(3)}")
print(f"Fields with all-zero NDVI: {(ndvi_traj.sum(axis=1) == 0).sum()}")

# ============================================================
# CLUSTER NDVI BASELINE
# ============================================================
print("\nClustering NDVI baseline...")

scaler = StandardScaler()
X_ndvi = scaler.fit_transform(ndvi_features)

km_ndvi = KMeans(n_clusters=5, random_state=42, n_init=10)
ndvi_labels = km_ndvi.fit_predict(X_ndvi)

unique_labels = np.unique(ndvi_labels)
print(f"Unique clusters: {len(unique_labels)}")
print(f"Cluster sizes: {np.bincount(ndvi_labels)}")

if len(unique_labels) >= 2:
    ndvi_sil = silhouette_score(X_ndvi, ndvi_labels)
    ndvi_db  = davies_bouldin_score(X_ndvi, ndvi_labels)
    print(f"Silhouette: {ndvi_sil:.4f}")
    print(f"Davies-Bouldin: {ndvi_db:.4f}")
else:
    print("ERROR: Only 1 cluster found — all NDVI features identical")
    ndvi_sil = 0.0
    ndvi_db  = 999.0

# ============================================================
# LOAD PRITHVI RESULTS
# ============================================================
print("\nLoading Prithvi embeddings...")
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    prithvi_results = pickle.load(f)
embeddings = np.stack([r["embedding"] for r in prithvi_results])

scaler2 = StandardScaler()
X_prithvi = scaler2.fit_transform(embeddings)
pca = PCA(n_components=50, random_state=42)
X_prithvi_pca = pca.fit_transform(X_prithvi)

km_prithvi = KMeans(n_clusters=5, random_state=42, n_init=10)
prithvi_labels = km_prithvi.fit_predict(X_prithvi_pca)
prithvi_sil = silhouette_score(X_prithvi_pca, prithvi_labels)
prithvi_db  = davies_bouldin_score(X_prithvi_pca, prithvi_labels)

# ============================================================
# RESULTS TABLE
# ============================================================
print("\n" + "="*60)
print("BENCHMARK RESULTS — Franconia Sugar Beet Fields, 2022")
print("n=1020 fields, 3 Sentinel-2 timesteps (Apr/Jun/Aug)")
print("="*60)
print(f"{'Model':<25} {'Silhouette':>12} {'Davies-Bouldin':>16}")
print("-"*60)
print(f"{'Prithvi EO 2.0':<25} {prithvi_sil:>12.4f} {prithvi_db:>16.4f}")
print(f"{'NDVI baseline':<25} {ndvi_sil:>12.4f} {ndvi_db:>16.4f}")
print("="*60)

better_sil = "Prithvi" if prithvi_sil > ndvi_sil else "NDVI"
better_db  = "Prithvi" if prithvi_db  < ndvi_db  else "NDVI"
print(f"\nBetter Silhouette:     {better_sil}")
print(f"Better Davies-Bouldin: {better_db}")

# ============================================================
# PLOT — NDVI TRAJECTORIES PER CLUSTER
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

for c in range(5):
    mask = ndvi_labels == c
    if mask.sum() == 0:
        continue
    traj = ndvi_traj[mask]
    mean_traj = traj.mean(axis=0)
    axes[0].plot(DATE_LABELS, mean_traj, 'o-',
                color=colors[c], linewidth=2,
                label=f"Cluster {c} (n={mask.sum()})")
axes[0].set_title('NDVI Baseline — Mean Trajectory per Cluster',
                  fontweight='bold')
axes[0].set_ylabel('Mean NDVI')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for c in range(5):
    mask = prithvi_labels == c
    if mask.sum() == 0:
        continue
    traj = ndvi_traj[mask]
    mean_traj = traj.mean(axis=0)
    axes[1].plot(DATE_LABELS, mean_traj, 'o-',
                color=colors[c], linewidth=2,
                label=f"Cluster {c} (n={mask.sum()})")
axes[1].set_title('Prithvi EO 2.0 — NDVI Trajectory per Cluster',
                  fontweight='bold')
axes[1].set_ylabel('Mean NDVI')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Benchmark: Prithvi EO 2.0 vs NDVI Baseline\n'
             'Sugar Beet Fields, Franconia 2022 (n=1020)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = f"{DATA_DIR}/benchmark_ndvi_trajectories.png"
plt.savefig(out_fig, dpi=120, bbox_inches='tight')
print(f"\nPlot saved: {out_fig}")

# Save
pd.DataFrame({
    "field_id":       [r["field_id"] for r in prithvi_results],
    "prithvi_cluster": prithvi_labels,
    "ndvi_cluster":   ndvi_labels,
    "ndvi_apr":       ndvi_traj[:, 0],
    "ndvi_jun":       ndvi_traj[:, 1],
    "ndvi_aug":       ndvi_traj[:, 2],
}).to_csv(f"{DATA_DIR}/benchmark_results_full.csv", index=False)
print("Results saved: Data/benchmark_results_full.csv")
