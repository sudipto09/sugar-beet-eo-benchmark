import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m")
from prithvi_mae import PrithviMAE

DATA_DIR  = "/home/sudiptochakraborty/praktikum_cv/Data"
MODEL_DIR = "/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m"

device = torch.device("cuda")
print(f"Device: {device}")


# 1. LOAD DATA

print("Loading embeddings and labels...")

with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl", "rb") as f:
    results = pickle.load(f)

embeddings = np.stack([r["embedding"] for r in results])  # (1020, 1536)

# Load NDVI features as supervision signal
df = pd.read_csv(f"{DATA_DIR}/benchmark_results_full.csv")

# NDVI trajectory as regression target - 3 values per field
ndvi_targets = np.column_stack([
    df["ndvi_apr"].values,
    df["ndvi_jun"].values,
    df["ndvi_aug"].values,
]).astype(np.float32)

print(f"Embeddings: {embeddings.shape}")
print(f"NDVI targets: {ndvi_targets.shape}")
print(f"NDVI range: {ndvi_targets.min():.3f} to {ndvi_targets.max():.3f}")

# Train/val split (80/20)
idx = np.arange(len(embeddings))
idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=42)
print(f"Train: {len(idx_train)}, Val: {len(idx_val)}")


# 2. DATASET

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets, indices):
        self.X = torch.tensor(embeddings[indices], dtype=torch.float32)
        self.y = torch.tensor(targets[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = EmbeddingDataset(embeddings, ndvi_targets, idx_train)
val_ds   = EmbeddingDataset(embeddings, ndvi_targets, idx_val)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)


# 3. FINE-TUNING HEAD

class NDVIHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)

head = NDVIHead().to(device)
optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()


# 4. TRAINING

print("\nFine-tuning NDVI prediction head on Prithvi embeddings...")
print("(Frozen Prithvi encoder + trainable MLP head)")

train_losses = []
val_losses   = []
best_val_loss = float('inf')
best_state = None

for epoch in range(60):
    # Train
    head.train()
    train_loss = 0.0
    for X_batch, y_batch in train_dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = head(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_ds)

    # Val
    head.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = head(X_batch)
            val_loss += criterion(pred, y_batch).item() * len(X_batch)
    val_loss /= len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

print(f"\nBest val loss: {best_val_loss:.6f}")


# 5. EXTRACT FINE-TUNED REPRESENTATIONS

head.load_state_dict(best_state)
head.eval()

class NDVIHeadWithHidden(nn.Module):
    
    def __init__(self, trained_head):
        super().__init__()
        self.layer1 = nn.Sequential(
            trained_head.net[0],
            trained_head.net[1],
            trained_head.net[2],
            trained_head.net[3],
        )
        self.layer2 = nn.Sequential(
            trained_head.net[4],
            trained_head.net[5],
        )
        self.out = trained_head.net[6]

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return h2  # 128-dim phenology-aware representation

extractor = NDVIHeadWithHidden(head).to(device)
extractor.eval()

print("\nExtracting fine-tuned representations...")
all_X = torch.tensor(embeddings, dtype=torch.float32)
finetuned_reps = []

with torch.no_grad():
    for i in range(0, len(all_X), 128):
        batch = all_X[i:i+128].to(device)
        rep = extractor(batch)
        finetuned_reps.append(rep.cpu().numpy())

finetuned_reps = np.vstack(finetuned_reps)
print(f"Fine-tuned representation shape: {finetuned_reps.shape}")


# 6. CLUSTER FINE-TUNED REPRESENTATIONS

print("\nClustering fine-tuned Prithvi representations...")

scaler = StandardScaler()
X_ft = scaler.fit_transform(finetuned_reps)

km_ft = KMeans(n_clusters=5, random_state=42, n_init=10)
ft_labels = km_ft.fit_predict(X_ft)
ft_sil = silhouette_score(X_ft, ft_labels)
ft_db  = davies_bouldin_score(X_ft, ft_labels)

print(f"Fine-tuned Prithvi:")
print(f"  Silhouette: {ft_sil:.4f}")
print(f"  Davies-Bouldin: {ft_db:.4f}")
print(f"  Cluster sizes: {np.bincount(ft_labels)}")


# 7. CHECK AGRONOMIC INTERPRETABILITY

ndvi_arr = ndvi_targets

print("\nFine-tuned cluster NDVI characteristics:")
for c in range(5):
    mask = ft_labels == c
    print(f"  Cluster {c} (n={mask.sum()}):"
          f"  Apr={ndvi_arr[mask,0].mean():.3f}"
          f"  Jun={ndvi_arr[mask,1].mean():.3f}"
          f"  Aug={ndvi_arr[mask,2].mean():.3f}"
          f"  Δ={ndvi_arr[mask,2].mean()-ndvi_arr[mask,1].mean():.3f}")


# 8. UPDATED BENCHMARK TABLE

# Reload other scores
df_bench = pd.read_csv(f"{DATA_DIR}/benchmark_results_complete.csv")

# Recompute for consistency
with open(f"{DATA_DIR}/prithvi_embeddings_all_fields.pkl","rb") as f:
    pr = pickle.load(f)
emb = np.stack([r["embedding"] for r in pr])
X_p = StandardScaler().fit_transform(emb)
X_p = PCA(50, random_state=42).fit_transform(X_p)
p_lab = KMeans(5, random_state=42, n_init=10).fit_predict(X_p)
p_sil = silhouette_score(X_p, p_lab)
p_db  = davies_bouldin_score(X_p, p_lab)

ndvi_feats = np.column_stack([
    df["ndvi_apr"].values,
    df["ndvi_jun"].values,
    df["ndvi_aug"].values,
    df["ndvi_aug"].values - df["ndvi_jun"].values,
    df["ndvi_aug"].values - df["ndvi_apr"].values,
]).astype(np.float32)
X_n = StandardScaler().fit_transform(ndvi_feats)
n_lab = KMeans(5, random_state=42, n_init=10).fit_predict(X_n)
n_sil = silhouette_score(X_n, n_lab)
n_db  = davies_bouldin_score(X_n, n_lab)


print("UPDATED BENCHMARK - Franconia Sugar Beet Fields 2022 (n=1020)")
print("="*68)
print(f"{'Model':<35} {'Silhouette ↑':>12} {'Davies-Bouldin ↓':>16}")
print("-"*68)
print(f"{'Prithvi EO 2.0 (zero-shot)':<35} {p_sil:>12.4f} {p_db:>16.4f}")
print(f"{'Prithvi EO 2.0 (fine-tuned head)':<35} {ft_sil:>12.4f} {ft_db:>16.4f}")
print(f"{'NDVI baseline':<35} {n_sil:>12.4f} {n_db:>16.4f}")
print(f"{'Random Forest (leaf embed)':<35} {0.1808:>12.4f} {1.9522:>16.4f}")
print(f"{'Raw Spectral KMeans':<35} {0.1793:>12.4f} {1.4884:>16.4f}")
print("="*68)


# 9. TRAINING CURVE PLOT

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train loss', color='#1f77b4')
axes[0].plot(val_losses,   label='Val loss',   color='#ff7f0e')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Fine-tuning: NDVI Prediction Loss\n(MLP head on frozen Prithvi)',
                  fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# NDVI trajectories per fine-tuned cluster
colors5 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for c in range(5):
    mask = ft_labels == c
    if mask.sum() == 0:
        continue
    mean_traj = ndvi_arr[mask].mean(axis=0)
    axes[1].plot(["Apr 16","Jun 18","Aug 7"], mean_traj,
                'o-', color=colors5[c], linewidth=2,
                label=f"Cluster {c} (n={mask.sum()})")
axes[1].set_title('Fine-tuned Prithvi - NDVI Trajectory per Cluster'
                  ,
                  fontweight='bold')
axes[1].set_ylabel('Mean NDVI')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Prithvi EO 2.0 Fine-tuning with NDVI Supervision\n'
             'Sugar Beet Fields, Franconia 2022',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/finetuned_prithvi_results.png",
            dpi=120, bbox_inches='tight')
print(f"\nPlot saved: Data/finetuned_prithvi_results.png")

# Save
pd.DataFrame({
    "field_id": [r["field_id"] for r in results],
    "finetuned_cluster": ft_labels,
}).to_csv(f"{DATA_DIR}/finetuned_cluster_assignments.csv", index=False)
print("Saved: Data/finetuned_cluster_assignments.csv")
