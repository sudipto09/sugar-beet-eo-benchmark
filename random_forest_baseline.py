import numpy as np
import pickle
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR  = "/home/sudiptochakraborty/praktikum_cv/Data"
SCENE_DIR = f"{DATA_DIR}/sentinel2/wuerzburg_core"
DATES     = ["20220416", "20220618", "20220807"]
DATE_LABELS = ["Apr 16", "Jun 18", "Aug 7"]
BANDS     = ["B02", "B03", "B04", "B05", "B06", "B07", "B08"]

# Load fields
gdf = gpd.read_file(f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson")
with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
    scene_crs = src.crs
gdf = gdf.to_crs(scene_crs)

# Load existing results
df_existing = pd.read_csv(f"{DATA_DIR}/benchmark_results_full.csv")

# ============================================================
# STEP 1: Extract full spectral features per field
# All 7 bands × 3 timesteps = 21 features per field
# This is what a classical RF approach uses
# ============================================================
print("Extracting full spectral features (7 bands × 3 dates)...")

spectral_features = []
valid_field_ids = []

for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
    field_geom = row.geometry
    field_id   = row["field_id"]
    field_feats = []
    valid = True

    for date in DATES:
        date_feats = []
        for band in BANDS:
            path = f"{SCENE_DIR}/{date}/{band}.tif"
            try:
                with rasterio.open(path) as src:
                    out, _ = rio_mask(src, [field_geom], crop=True,
                                      nodata=0, all_touched=True)
                    out = out[0].astype(np.float32)
                valid_px = out[out > 100]
                if len(valid_px) > 0:
                    date_feats.append(float(valid_px.mean()))
                else:
                    date_feats.append(0.0)
            except Exception:
                date_feats.append(0.0)
        field_feats.extend(date_feats)

    spectral_features.append(field_feats)
    valid_field_ids.append(field_id)

spectral_features = np.array(spectral_features)
print(f"Spectral features shape: {spectral_features.shape}")  # (1020, 21)
print(f"Feature range: {spectral_features.min():.1f} to {spectral_features.max():.1f}")
print(f"Fields with all-zero features: {(spectral_features.sum(axis=1) == 0).sum()}")

# ============================================================
# STEP 2: Raw spectral KMeans (classical baseline)
# ============================================================
print("\nClustering raw spectral features (KMeans)...")

scaler = StandardScaler()
X_spectral = scaler.fit_transform(spectral_features)

km_spectral = KMeans(n_clusters=5, random_state=42, n_init=10)
spectral_labels = km_spectral.fit_predict(X_spectral)
spectral_sil = silhouette_score(X_spectral, spectral_labels)
spectral_db  = davies_bouldin_score(X_spectral, spectral_labels)

print(f"Spectral KMeans:")
print(f"  Silhouette: {spectral_sil:.4f}")
print(f"  Davies-Bouldin: {spectral_db:.4f}")
print(f"  Cluster sizes: {np.bincount(spectral_labels)}")

# ============================================================
# STEP 3: Random Forest leaf embeddings
# Use NDVI cluster labels as pseudo-labels to train RF
# Then extract leaf node activations as RF embedding
# This is the standard "RF as feature extractor" approach
# ============================================================
print("\nTraining Random Forest on NDVI pseudo-labels...")

# Use NDVI cluster labels as pseudo-supervision
ndvi_labels = df_existing["ndvi_cluster"].values

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_spectral, ndvi_labels)

# Cross-validation score — how well does RF predict NDVI clusters?
cv_scores = cross_val_score(rf, X_spectral, ndvi_labels, cv=5, scoring='accuracy')
print(f"RF cross-val accuracy predicting NDVI clusters: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Extract RF leaf embeddings (proximity matrix approach)
# apply() returns leaf node index for each tree per sample
leaf_nodes = rf.apply(X_spectral)  # shape: (1020, n_estimators)
print(f"RF leaf embedding shape: {leaf_nodes.shape}")

# Reduce leaf embeddings for clustering
# One-hot encode leaf nodes then PCA
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
leaf_encoded = enc.fit_transform(leaf_nodes)
print(f"Leaf encoded shape: {leaf_encoded.shape}")

pca_rf = PCA(n_components=50, random_state=42)
X_rf = pca_rf.fit_transform(leaf_encoded)
print(f"PCA variance explained: {pca_rf.explained_variance_ratio_.sum():.2%}")

# Cluster RF embeddings
km_rf = KMeans(n_clusters=5, random_state=42, n_init=10)
rf_labels = km_rf.fit_predict(X_rf)
rf_sil = silhouette_score(X_rf, rf_labels)
rf_db  = davies_bouldin_score(X_rf, rf_labels)

print(f"\nRandom Forest (leaf embeddings):")
print(f"  Silhouette: {rf_sil:.4f}")
print(f"  Davies-Bouldin: {rf_db:.4f}")
print(f"  Cluster sizes: {np.bincount(rf_labels)}")

# ============================================================
# STEP 4: FEATURE IMPORTANCE — what does RF care about?
# ============================================================
feature_names = []
for date in DATE_LABELS:
    for band in BANDS:
        feature_names.append(f"{band}_{date}")

importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]

print("\nTop 10 most important features (RF):")
for i in top_idx:
    print(f"  {feature_names[i]:<20}: {importances[i]:.4f}")

# ============================================================
# STEP 5: COMPLETE BENCHMARK TABLE
# ============================================================
# Load Prithvi scores
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    prithvi_results = pickle.load(f)
embeddings = np.stack([r["embedding"] for r in prithvi_results])
scaler2 = StandardScaler()
X_prithvi = scaler2.fit_transform(embeddings)
pca2 = PCA(n_components=50, random_state=42)
X_prithvi_pca = pca2.fit_transform(X_prithvi)
km_p = KMeans(n_clusters=5, random_state=42, n_init=10)
prithvi_labels = km_p.fit_predict(X_prithvi_pca)
prithvi_sil = silhouette_score(X_prithvi_pca, prithvi_labels)
prithvi_db  = davies_bouldin_score(X_prithvi_pca, prithvi_labels)

# NDVI scores
ndvi_feats = np.column_stack([
    df_existing["ndvi_apr"].values,
    df_existing["ndvi_jun"].values,
    df_existing["ndvi_aug"].values,
    df_existing["ndvi_aug"].values - df_existing["ndvi_jun"].values,
    df_existing["ndvi_aug"].values - df_existing["ndvi_apr"].values,
])
X_ndvi = StandardScaler().fit_transform(ndvi_feats)
ndvi_labels_2 = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_ndvi)
ndvi_sil = silhouette_score(X_ndvi, ndvi_labels_2)
ndvi_db  = davies_bouldin_score(X_ndvi, ndvi_labels_2)

print("\n" + "="*65)
print("COMPLETE BENCHMARK — Franconia Sugar Beet Fields 2022 (n=1020)")
print("3 Sentinel-2 timesteps: Apr 16 / Jun 18 / Aug 7")
print("="*65)
print(f"{'Model':<30} {'Silhouette ↑':>13} {'Davies-Bouldin ↓':>17}")
print("-"*65)
print(f"{'Prithvi EO 2.0':<30} {prithvi_sil:>13.4f} {prithvi_db:>17.4f}")
print(f"{'Random Forest (leaf embed)':<30} {rf_sil:>13.4f} {rf_db:>17.4f}")
print(f"{'Raw Spectral KMeans':<30} {spectral_sil:>13.4f} {spectral_db:>17.4f}")
print(f"{'NDVI baseline':<30} {ndvi_sil:>13.4f} {ndvi_db:>17.4f}")
print("="*65)

# ============================================================
# STEP 6: PLOT — Feature importance bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature importance
top_names = [feature_names[i] for i in top_idx]
top_vals  = [importances[i] for i in top_idx]
axes[0].barh(top_names[::-1], top_vals[::-1], color='#2ca02c')
axes[0].set_title('Random Forest Feature Importance\n(predicting NDVI cluster labels)',
                  fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].grid(True, alpha=0.3, axis='x')

# Benchmark comparison bar chart
models = ['Prithvi\nEO 2.0', 'RF\n(leaf embed)', 'Raw Spectral\nKMeans', 'NDVI\nBaseline']
sil_scores = [prithvi_sil, rf_sil, spectral_sil, ndvi_sil]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

bars = axes[1].bar(models, sil_scores, color=colors, alpha=0.85, width=0.5)
axes[1].set_title('Silhouette Score Comparison\n(higher = better)',
                  fontweight='bold')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_ylim(0, max(sil_scores) * 1.3)
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, sil_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Benchmark: EO Foundation Model vs Classical Baselines\n'
             'Sugar Beet Fields, Franconia 2022 (n=1020)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/benchmark_complete.png", dpi=120, bbox_inches='tight')
print(f"\nPlot saved: Data/benchmark_complete.png")

# Save full results
pd.DataFrame({
    "field_id":          valid_field_ids,
    "prithvi_cluster":   prithvi_labels,
    "rf_cluster":        rf_labels,
    "spectral_cluster":  spectral_labels,
    "ndvi_cluster":      ndvi_labels_2,
}).to_csv(f"{DATA_DIR}/benchmark_results_complete.csv", index=False)
print("Complete results saved: Data/benchmark_results_complete.csv")
