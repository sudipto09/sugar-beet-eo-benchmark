import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import rasterio
from rasterio.mask import mask as rio_mask
from skimage.transform import resize
import geopandas as gpd
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m")
from prithvi_mae import PrithviMAE

DATA_DIR   = "/home/sudiptochakraborty/praktikum_cv/Data"
MODEL_DIR  = "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m"
SCENE_DIR  = f"{DATA_DIR}/sentinel2/wuerzburg_core_2021"
FIELDS_PATH = f"{DATA_DIR}/sugar_beet_fields_franconia_hires.geojson"

DATES  = ["20210426","20210618","20210814"]
BANDS_PRITHVI = ["B02","B03","B04","B05","B06","B07"]
BANDS_ALL     = ["B02","B03","B04","B05","B06","B07","B08"]
MEAN = np.array([1087.0,1342.0,1433.0,2734.0,1958.0,1363.0])
STD  = np.array([2248.0,2179.0,2178.0,1850.0,1242.0,1049.0])
IMG_SIZE = 64
N_CLUSTERS = 5
N_SEEDS = 10

device = torch.device("cuda")
print(f"Device: {device}")

# ============================================================
# LOAD FIELDS
# ============================================================
gdf = gpd.read_file(FIELDS_PATH)
with rasterio.open(f"{SCENE_DIR}/{DATES[0]}/B02.tif") as src:
    scene_crs = src.crs
gdf = gdf.to_crs(scene_crs)
print(f"Fields: {len(gdf)}")

# ============================================================
# STEP 1: EXTRACT PRITHVI EMBEDDINGS FOR 2021
# ============================================================
print("\nLoading Prithvi model...")
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)["pretrained_cfg"]
cfg.update(num_frames=3, in_chans=6, coords_encoding=[])

model = PrithviMAE(**cfg)
state_dict = torch.load(f"{MODEL_DIR}/Prithvi_EO_V2_300M.pt",
                        map_location=device, weights_only=True)
for k in list(state_dict.keys()):
    if 'pos_embed' in k: del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

def crop_band(date, band, field_geom):
    path = f"{SCENE_DIR}/{date}/{band}.tif"
    with rasterio.open(path) as src:
        out, _ = rio_mask(src,[field_geom],crop=True,nodata=0,all_touched=True)
    out = out[0].astype(np.float32)
    return None if (out.shape[0]<2 or out.shape[1]<2) else out

print("Extracting 2021 Prithvi embeddings...")
embeddings_2021 = []
ndvi_apr, ndvi_jun, ndvi_aug = [], [], []
spectral_feats = []
failed = 0

for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
    geom = row.geometry
    band_stacks = []
    valid = True

    # Prithvi embedding
    for b_idx, band in enumerate(BANDS_PRITHVI):
        frames = []
        for date in DATES:
            try:
                p = crop_band(date, band, geom)
            except: p = None
            if p is None: valid=False; break
            p = (p - MEAN[b_idx]) / STD[b_idx]
            p = resize(p,(IMG_SIZE,IMG_SIZE),
                      anti_aliasing=True,preserve_range=True).astype(np.float32)
            frames.append(p)
        if not valid: break
        band_stacks.append(np.stack(frames,axis=0))

    if not valid or len(band_stacks)!=6:
        failed+=1
        embeddings_2021.append(np.zeros(1536))
        ndvi_apr.append(0.0); ndvi_jun.append(0.0); ndvi_aug.append(0.0)
        spectral_feats.append([0.0]*21)
        continue

    x = torch.tensor(np.stack(band_stacks,0)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred, _ = model(x, None, None, 0.0)
        emb = pred.mean(dim=1).squeeze(0).cpu().numpy()
    embeddings_2021.append(emb)

    # NDVI per timestep
    ndvi_vals = []
    for t_idx, date in enumerate(DATES):
        try:
            b04 = crop_band(date,"B04",geom)
            b08 = crop_band(date,"B08",geom)
            if b04 is None or b08 is None:
                ndvi_vals.append(0.0)
            else:
                v = (b04>100)&(b08>100)
                ndvi_vals.append(float(((b08-b04)/(b08+b04+1e-8))[v].mean()) if v.sum()>0 else 0.0)
        except: ndvi_vals.append(0.0)
    ndvi_apr.append(ndvi_vals[0])
    ndvi_jun.append(ndvi_vals[1])
    ndvi_aug.append(ndvi_vals[2])

    # Spectral features (all 7 bands × 3 dates)
    sf = []
    for date in DATES:
        for band in BANDS_ALL:
            try:
                p = crop_band(date, band, geom)
                px = p[p>100] if p is not None else np.array([])
                sf.append(float(px.mean()) if len(px)>0 else 0.0)
            except: sf.append(0.0)
    spectral_feats.append(sf)

embeddings_2021 = np.array(embeddings_2021)
ndvi_traj_2021  = np.column_stack([ndvi_apr, ndvi_jun, ndvi_aug]).astype(np.float32)
spectral_2021   = np.array(spectral_feats)

print(f"Embeddings: {embeddings_2021.shape}, failed: {failed}")
print(f"2021 NDVI range: {ndvi_traj_2021[ndvi_traj_2021>0].min():.3f} to {ndvi_traj_2021.max():.3f}")
print(f"Mean NDVI per timestep: {ndvi_traj_2021.mean(axis=0).round(3)}")

# ============================================================
# STEP 2: NDVI FEATURES
# ============================================================
ndvi_feats_2021 = np.column_stack([
    ndvi_traj_2021,
    ndvi_traj_2021[:,2] - ndvi_traj_2021[:,1],
    ndvi_traj_2021[:,2] - ndvi_traj_2021[:,0],
]).astype(np.float32)

# ============================================================
# STEP 3: FINE-TUNING FUNCTION (same as 2022)
# ============================================================
class NDVIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.GELU(),
            nn.Linear(128,3),
        )
    def forward(self,x): return self.net(x)

class HeadExtractor(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.layers = nn.Sequential(*list(head.net.children())[:6])
    def forward(self,x): return self.layers(x)

def finetune_embed(embeddings, targets, seed):
    torch.manual_seed(seed)
    idx_tr,_ = train_test_split(np.arange(len(embeddings)),test_size=0.2,random_state=seed)
    X_t = torch.tensor(embeddings[idx_tr],dtype=torch.float32)
    y_t = torch.tensor(targets[idx_tr],   dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t,y_t),batch_size=64,shuffle=True)
    head = NDVIHead().to(device)
    opt  = optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    crit = nn.MSELoss()
    for _ in range(60):
        head.train()
        for xb,yb in dl:
            xb,yb = xb.to(device),yb.to(device)
            opt.zero_grad(); crit(head(xb),yb).backward(); opt.step()
        sch.step()
    ext = HeadExtractor(head).to(device).eval()
    reps=[]
    with torch.no_grad():
        for i in range(0,len(embeddings),128):
            reps.append(ext(torch.tensor(embeddings[i:i+128],dtype=torch.float32).to(device)).cpu().numpy())
    return np.vstack(reps)

def iqm(arr):
    arr=np.array(arr)
    q25,q75=np.percentile(arr,[25,75])
    return float(np.mean(arr[(arr>=q25)&(arr<=q75)]))

def bootstrap_ci(arr,n=1000,ci=0.95):
    arr=np.array(arr)
    m=[np.mean(np.random.choice(arr,len(arr),replace=True)) for _ in range(n)]
    return np.percentile(m,(1-ci)/2*100), np.percentile(m,(1+ci)/2*100)

# Pre-compute scaled features
X_prithvi_2021 = PCA(50,random_state=0).fit_transform(
    StandardScaler().fit_transform(embeddings_2021))
X_ndvi_2021    = StandardScaler().fit_transform(ndvi_feats_2021)
X_spec_2021    = StandardScaler().fit_transform(spectral_2021)

# ============================================================
# STEP 4: 10-SEED EVALUATION
# ============================================================
print(f"\nRunning {N_SEEDS} seeds on 2021 data...")

MODELS = {
    "Prithvi (zero-shot)":  {"X": X_prithvi_2021, "ft": False},
    "NDVI baseline":        {"X": X_ndvi_2021,    "ft": False},
    "Raw Spectral KMeans":  {"X": X_spec_2021,    "ft": False},
    "Prithvi (fine-tuned)": {"X": None,           "ft": True},
}

scores_2021 = {n:{"sil":[],"db":[]} for n in MODELS}

for seed in range(N_SEEDS):
    print(f"  Seed {seed+1}/{N_SEEDS}...")
    for name,cfg2 in MODELS.items():
        if cfg2["ft"]:
            reps = finetune_embed(embeddings_2021, ndvi_traj_2021, seed)
            X = StandardScaler().fit_transform(reps)
        else:
            X = cfg2["X"]
        km = KMeans(N_CLUSTERS,random_state=seed,n_init=10)
        lab = km.fit_predict(X)
        scores_2021[name]["sil"].append(silhouette_score(X,lab))
        scores_2021[name]["db"].append(davies_bouldin_score(X,lab))

# ============================================================
# STEP 5: CROSS-YEAR COMPARISON TABLE
# ============================================================
# Load 2022 scores
df_seeds = pd.read_csv(f"{DATA_DIR}/all_seeds_results.csv")
scores_2022 = {}
for name in MODELS:
    sub = df_seeds[df_seeds["model"]==name]
    scores_2022[name] = {
        "sil": sub["silhouette"].tolist(),
        "db":  sub["davies_bouldin"].tolist()
    }

print("\n" + "="*85)
print("CROSS-YEAR BENCHMARK — 2022 vs 2021 (10 seeds, IQM)")
print("Sugar Beet Fields, Franconia (n=1020)")
print("="*85)
print(f"{'Model':<30} {'2022 Sil':>10} {'2021 Sil':>10} {'2022 DB':>10} {'2021 DB':>10} {'Rank stable?':>14}")
print("-"*85)

rank_2022 = sorted(MODELS.keys(),
                   key=lambda n: iqm(scores_2022[n]["sil"]), reverse=True)
rank_2021 = sorted(MODELS.keys(),
                   key=lambda n: iqm(scores_2021[n]["sil"]), reverse=True)

for name in rank_2022:
    s22 = iqm(scores_2022[name]["sil"])
    s21 = iqm(scores_2021[name]["sil"])
    d22 = iqm(scores_2022[name]["db"])
    d21 = iqm(scores_2021[name]["db"])
    stable = "✓" if rank_2021.index(name)==rank_2022.index(name) else "✗"
    print(f"{name:<30} {s22:>10.4f} {s21:>10.4f} {d22:>10.4f} {d21:>10.4f} {stable:>14}")

print("="*85)
print(f"2022 ranking: {' > '.join(rank_2022)}")
print(f"2021 ranking: {' > '.join(rank_2021)}")
stable = rank_2022 == rank_2021
print(f"Ranking fully stable across years: {stable}")

# Save 2021 scores
rows=[]
for name in MODELS:
    for seed in range(N_SEEDS):
        rows.append({"model":name,"year":2021,"seed":seed,
                     "silhouette":scores_2021[name]["sil"][seed],
                     "davies_bouldin":scores_2021[name]["db"][seed]})
pd.DataFrame(rows).to_csv(f"{DATA_DIR}/all_seeds_2021.csv",index=False)
print("\nSaved: Data/all_seeds_2021.csv")
