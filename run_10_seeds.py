import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/home/sudiptochakraborty/praktikum_cv/Data"
N_SEEDS  = 10
N_CLUSTERS = 5

print("Loading data...")
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    results = pickle.load(f)
embeddings = np.stack([r["embedding"] for r in results])

df = pd.read_csv(f"{DATA_DIR}/benchmark_results_full.csv")
ndvi_feats = np.column_stack([
    df["ndvi_apr"].values,
    df["ndvi_jun"].values,
    df["ndvi_aug"].values,
    df["ndvi_aug"].values - df["ndvi_jun"].values,
    df["ndvi_aug"].values - df["ndvi_apr"].values,
]).astype(np.float32)

df_bench = pd.read_csv(f"{DATA_DIR}/benchmark_results_complete.csv")
spectral_path = f"{DATA_DIR}/spectral_features.npy"

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from tqdm import tqdm

SCENE_DIR = f"{DATA_DIR}/sentinel2/wuerzburg_core"
DATES     = ["20220416","20220618","20220807"]
BANDS     = ["B02","B03","B04","B05","B06","B07","B08"]

import os
if not os.path.exists(spectral_path):
    print("Computing spectral features...")
    gdf = gpd.read_file(f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson")
    with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
        gdf = gdf.to_crs(src.crs)
    feats = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
        f = []
        for date in DATES:
            for band in BANDS:
                try:
                    with rasterio.open(f"{SCENE_DIR}/{date}/{band}.tif") as src:
                        out, _ = rio_mask(src, [row.geometry], crop=True,
                                         nodata=0, all_touched=True)
                        px = out[0][out[0] > 100].astype(np.float32)
                        f.append(float(px.mean()) if len(px)>0 else 0.0)
                except:
                    f.append(0.0)
        feats.append(f)
    spectral = np.array(feats)
    np.save(spectral_path, spectral)
    print(f"Saved spectral features: {spectral.shape}")
else:
    spectral = np.load(spectral_path)
    print(f"Loaded spectral features: {spectral.shape}")

ndvi_targets = np.column_stack([
    df["ndvi_apr"].values,
    df["ndvi_jun"].values,
    df["ndvi_aug"].values,
]).astype(np.float32)

device = torch.device("cuda")

# ============================================================
# FINE-TUNING FUNCTION
# ============================================================
class NDVIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )
    def forward(self, x): return self.net(x)

class HeadExtractor(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.layers = nn.Sequential(*list(head.net.children())[:6])
    def forward(self, x): return self.layers(x)

def finetune_and_embed(embeddings, targets, seed):
    torch.manual_seed(seed)
    idx_tr, idx_val = train_test_split(
        np.arange(len(embeddings)), test_size=0.2, random_state=seed
    )
    X_t = torch.tensor(embeddings[idx_tr], dtype=torch.float32)
    y_t = torch.tensor(targets[idx_tr],    dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    head = NDVIHead().to(device)
    opt  = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    crit = nn.MSELoss()

    for _ in range(60):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        sch.step()

    ext = HeadExtractor(head).to(device).eval()
    all_x = torch.tensor(embeddings, dtype=torch.float32)
    reps = []
    with torch.no_grad():
        for i in range(0, len(all_x), 128):
            reps.append(ext(all_x[i:i+128].to(device)).cpu().numpy())
    return np.vstack(reps)

# ============================================================
# MULTI-SEED EVALUATION
# ============================================================
def cluster_scores(X, seed, n=N_CLUSTERS):
    km = KMeans(n_clusters=n, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return (silhouette_score(X, labels),
            davies_bouldin_score(X, labels))

def iqm(arr):
    arr = np.array(arr)
    q25, q75 = np.percentile(arr, [25, 75])
    trimmed = arr[(arr >= q25) & (arr <= q75)]
    return float(np.mean(trimmed))

def bootstrap_ci(arr, n=1000, ci=0.95):
    arr = np.array(arr)
    means = [np.mean(np.random.choice(arr, len(arr), replace=True))
             for _ in range(n)]
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return lo, hi

# Pre-compute scaled features (deterministic)
scaler_p = StandardScaler()
X_prithvi = PCA(50, random_state=0).fit_transform(
    scaler_p.fit_transform(embeddings))

scaler_n = StandardScaler()
X_ndvi = scaler_n.fit_transform(ndvi_feats)

scaler_s = StandardScaler()
X_spec = scaler_s.fit_transform(spectral)

MODELS = {
    "Prithvi (zero-shot)":  {"X": X_prithvi,  "finetune": False},
    "NDVI baseline":        {"X": X_ndvi,      "finetune": False},
    "Raw Spectral KMeans":  {"X": X_spec,      "finetune": False},
    "Prithvi (fine-tuned)": {"X": None,        "finetune": True},
}

all_scores = {name: {"sil": [], "db": []} for name in MODELS}

print(f"\nRunning {N_SEEDS} seeds for each model...")
for seed in range(N_SEEDS):
    print(f"  Seed {seed+1}/{N_SEEDS}...")
    for name, cfg in MODELS.items():
        if cfg["finetune"]:
            reps = finetune_and_embed(embeddings, ndvi_targets, seed)
            X = StandardScaler().fit_transform(reps)
        else:
            X = cfg["X"]
        sil, db = cluster_scores(X, seed)
        all_scores[name]["sil"].append(sil)
        all_scores[name]["db"].append(db)

# ============================================================
# RESULTS WITH IQM + BOOTSTRAPPED CI
# ============================================================
print("\n" + "="*80)
print("GEO-BENCH STYLE RESULTS — 10 seeds, IQM + 95% bootstrapped CI")
print("Franconia Sugar Beet Fields 2022, n=1020")
print("="*80)
print(f"{'Model':<30} {'Sil IQM':>10} {'Sil 95% CI':>20} {'DB IQM':>10} {'DB 95% CI':>20}")
print("-"*80)

final_scores = {}
for name in MODELS:
    sil_arr = all_scores[name]["sil"]
    db_arr  = all_scores[name]["db"]
    sil_iqm = iqm(sil_arr)
    db_iqm  = iqm(db_arr)
    sil_ci  = bootstrap_ci(sil_arr)
    db_ci   = bootstrap_ci(db_arr)
    final_scores[name] = {
        "sil_iqm": sil_iqm, "sil_ci": sil_ci,
        "db_iqm": db_iqm,   "db_ci": db_ci,
        "sil_all": sil_arr, "db_all": db_arr
    }
    print(f"{name:<30} {sil_iqm:>10.4f} "
          f"[{sil_ci[0]:.4f}, {sil_ci[1]:.4f}]"
          f"  {db_iqm:>10.4f} "
          f"[{db_ci[0]:.4f}, {db_ci[1]:.4f}]")

print("="*80)

# ============================================================
# VIOLIN PLOT (GEO-Bench Fig 4 style)
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names  = list(MODELS.keys())
colors = ['#1f77b4','#2ca02c','#ff7f0e','#9467bd']

for ax, metric, label, better in zip(
    axes,
    ["sil", "db"],
    ["Silhouette Score (higher = better)",
     "Davies-Bouldin Score (lower = better)"],
    ["↑", "↓"]
):
    data = [final_scores[n][f"{metric}_all"] for n in names]
    vp = ax.violinplot(data, positions=range(len(names)),
                       showmedians=True, showextrema=True)
    for i, (body, c) in enumerate(zip(vp["bodies"], colors)):
        body.set_facecolor(c)
        body.set_alpha(0.7)

    for i, (n, c) in enumerate(zip(names, colors)):
        iq = final_scores[n][f"{metric}_iqm"]
        ax.scatter(i, iq, color=c, s=80, zorder=5,
                   marker='D', label=f"{n}: IQM={iq:.3f}")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names],
                       fontsize=9)
    ax.set_ylabel(label)
    ax.set_title(f"{label} {better}\n(10 seeds, IQM = diamond)",
                 fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("GEO-Bench Style Benchmark\n"
             "Prithvi EO 2.0 vs Baselines — Sugar Beet Franconia 2022",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/geobench_style_results.png",
            dpi=120, bbox_inches='tight')
print(f"\nPlot saved: Data/geobench_style_results.png")

# Save raw seed scores
rows = []
for name in MODELS:
    for seed in range(N_SEEDS):
        rows.append({
            "model": name,
            "seed": seed,
            "silhouette": all_scores[name]["sil"][seed],
            "davies_bouldin": all_scores[name]["db"][seed],
        })
pd.DataFrame(rows).to_csv(
    f"{DATA_DIR}/all_seeds_results.csv", index=False)
print("All seed scores saved: Data/all_seeds_results.csv")
