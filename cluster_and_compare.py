import numpy as np
import pickle
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = "/home/sudiptochakraborty/praktikum_cv/Data"
SCENE_DIR  = f"{DATA_DIR}/sentinel2/wuerzburg_core"
DATES      = ["20220416", "20220618", "20220807"]
DATE_LABELS = ["Apr 16", "Jun 18", "Aug 7"]

# ============================================================
# 1. LOAD PRITHVI EMBEDDINGS
# ============================================================
print("Loading Prithvi embeddings...")
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    results = pickle.load(f)

field_ids  = [r["field_id"]  for r in results]
embeddings = np.stack([r["embedding"] for r in results])
print(f"Embeddings: {embeddings.shape}")  # (1020, 1536)

# ============================================================
# 2. COMPUTE NDVI / NDRE BASELINE PER FIELD
# ============================================================
print("Computing NDVI/NDRE baseline per field...")

gdf = gpd.read_file(f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson")
with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
    scene_crs = src.crs
gdf = gdf.to_crs(scene_crs)

ndvi_features = []

for idx, row in gdf.iterrows():
    field_geom = row.geometry
    field_ndvi = []
    field_ndre = []

    for date in DATES:
        try:
            with rasterio.open(f"{SCENE_DIR}/{date}/B04.tif") as src:
                b04, _ = rio_mask(src, [field_geom], crop=True, nodata=0, all_touched=True)
                b04 = b04[0].astype(np.float32)

            with rasterio.open(f"{SCENE_DIR}/{date}/B07.tif") as src:
                b07, _ = rio_mask(src, [field_geom], crop=True, nodata=0, all_touched=True)
                b07 = b07[0].astype(np.float32)

            with rasterio.open(f"{SCENE_DIR}/{date}/B05.tif") as src:
                b05, _ = rio_mask(src, [field_geom], crop=True, nodata=0, all_touched=True)
                b05 = b05[0].astype(np.float32)

            eps = 1e-8
            # NDVI: (B07-B04)/(B07+B04) — using B07 as NIR proxy
            ndvi = (b07 - b04) / (b07 + b04 + eps)
            valid = (b04 > 0) & (b07 > 0)
            field_ndvi.append(float(ndvi[valid].mean()) if valid.sum() > 0 else 0.0)

            # NDRE: (B07-B05)/(B07+B05)
            ndre = (b07 - b05) / (b07 + b05 + eps)
            valid2 = (b07 > 0) & (b05 > 0)
            field_ndre.append(float(ndre[valid2].mean()) if valid2.sum() > 0 else 0.0)

        except Exception:
            field_ndvi.append(0.0)
            field_ndre.append(0.0)

    # Feature vector: [NDVI_t1, NDVI_t2, NDVI_t3, NDRE_t1, NDRE_t2, NDRE_t3]
    ndvi_features.append(field_ndvi + field_ndre)

ndvi_features = np.array(ndvi_features)
print(f"NDVI/NDRE features: {ndvi_features.shape}")  # (1020, 6)

# ============================================================
# 3. CLUSTERING — PRITHVI vs NDVI BASELINE
# ============================================================
print("\nClustering...")

N_CLUSTERS = 5

def cluster_and_score(features, name, n_clusters=N_CLUSTERS):
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # PCA for high-dim features
    if X.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        X = pca.fit_transform(X)
        print(f"  {name}: PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Scores
    sil  = silhouette_score(X, labels)
    db   = davies_bouldin_score(X, labels)

    print(f"  {name}:")
    print(f"    Silhouette score:     {sil:.4f}  (higher = better, max 1.0)")
    print(f"    Davies-Bouldin score: {db:.4f}  (lower = better)")
    print(f"    Cluster sizes: {np.bincount(labels)}")

    return labels, X, sil, db

print("\n--- PRITHVI EO 2.0 ---")
prithvi_labels, prithvi_X, prithvi_sil, prithvi_db = cluster_and_score(
    embeddings, "Prithvi EO 2.0"
)

print("\n--- NDVI/NDRE BASELINE ---")
ndvi_labels, ndvi_X, ndvi_sil, ndvi_db = cluster_and_score(
    ndvi_features, "NDVI/NDRE baseline"
)

# ============================================================
# 4. RESULTS TABLE
# ============================================================
print("\n" + "="*55)
print("BENCHMARK RESULTS — Franconia Sugar Beet Fields 2022")
print("="*55)
print(f"{'Model':<25} {'Silhouette':>12} {'Davies-Bouldin':>16}")
print("-"*55)
print(f"{'Prithvi EO 2.0':<25} {prithvi_sil:>12.4f} {prithvi_db:>16.4f}")
print(f"{'NDVI/NDRE baseline':<25} {ndvi_sil:>12.4f} {ndvi_db:>16.4f}")
print("="*55)

# ============================================================
# 5. VISUALIZE — PCA scatter plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prithvi
pca2 = PCA(n_components=2, random_state=42)
prithvi_2d = pca2.fit_transform(prithvi_X)
scatter = axes[0].scatter(prithvi_2d[:, 0], prithvi_2d[:, 1],
                          c=prithvi_labels, cmap='tab10', alpha=0.6, s=20)
axes[0].set_title(f'Prithvi EO 2.0 Clusters\n'
                  f'Silhouette={prithvi_sil:.3f}, DB={prithvi_db:.3f}',
                  fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[0])

# NDVI baseline
pca2b = PCA(n_components=2, random_state=42)
ndvi_2d = pca2b.fit_transform(ndvi_X)
scatter2 = axes[1].scatter(ndvi_2d[:, 0], ndvi_2d[:, 1],
                           c=ndvi_labels, cmap='tab10', alpha=0.6, s=20)
axes[1].set_title(f'NDVI/NDRE Baseline Clusters\n'
                  f'Silhouette={ndvi_sil:.3f}, DB={ndvi_db:.3f}',
                  fontweight='bold')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[1])

plt.suptitle('Benchmark: Prithvi EO 2.0 vs NDVI Baseline\n'
             'Sugar Beet Fields, Franconia 2022 (n=1020)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/benchmark_results_2022.png", dpi=120, bbox_inches='tight')
print(f"\nPlot saved: Data/benchmark_results_2022.png")

# Save results
results_df = pd.DataFrame({
    "field_id": field_ids,
    "prithvi_cluster": prithvi_labels,
    "ndvi_cluster": ndvi_labels,
})
results_df.to_csv(f"{DATA_DIR}/cluster_assignments_2022.csv", index=False)
print("Results saved: Data/cluster_assignments_2022.csv")
