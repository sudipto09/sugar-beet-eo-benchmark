"""Make README figures from results/*.csv. Usage: python make_figures.py"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R, F = Path("results"), Path("figures"); F.mkdir(exist_ok=True)
TARGETS = ["ndvi_greenup", "ndvi_senescence", "ndvi_amplitude"]

# Fig 1: layer-wise probe R^2 (per-layer rows have a numeric 'layer')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, yr in zip(axes, (2021, 2022)):
    d = pd.read_csv(R / f"probe_layerwise_{yr}.csv")
    d = d[d["layer"].notna()].sort_values("layer")
    for t in TARGETS:
        ax.plot(d["layer"], d[t], marker="o", ms=3, label=t.replace("ndvi_", ""))
    ax.set_title(str(yr)); ax.set_xlabel("encoder block"); ax.grid(alpha=.3)
axes[0].set_ylabel("probe $R^2$"); axes[0].legend(frameon=False)
fig.suptitle("Layer-wise probing of frozen Prithvi EO 2.0")
fig.tight_layout(); fig.savefig(F / "layerwise.png", dpi=150); plt.close(fig)

# Fig 2: held-out date R^2 with bootstrap CIs
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
for ax, yr in zip(axes, (2021, 2022)):
    d = pd.read_csv(R / f"probe_heldout_{yr}.csv")
    d = d[d["model"] == "ridge"]
    y = np.arange(len(d))[::-1]
    ax.errorbar(d["r2"], y, xerr=[d["r2"] - d["lo"], d["hi"] - d["r2"]], fmt="o", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(d["features"], fontsize=8)
    ax.set_title(str(yr)); ax.set_xlabel("held-out date $R^2$ (95% CI)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F / "heldout.png", dpi=150); plt.close(fig)

# Fig 3: clustering silhouette, 2022
d = pd.read_csv(R / "benchmark_v2_2022_bootstrap.csv")
fig, ax = plt.subplots(figsize=(6, 3))
y = np.arange(len(d))[::-1]
ax.errorbar(d["silhouette_mean"], y,
            xerr=[d["silhouette_mean"] - d["silhouette_lo"], d["silhouette_hi"] - d["silhouette_mean"]],
            fmt="o", capsize=3)
ax.set_yticks(y); ax.set_yticklabels(d["model"], fontsize=8)
ax.set_xlabel("silhouette (higher is better, 95% CI)"); ax.set_title("Clustering, 2022"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F / "clustering.png", dpi=150); plt.close(fig)