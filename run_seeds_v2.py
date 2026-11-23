"""
Multi-seed clustering benchmark with intervals that mean something.

What was wrong with run_10_seeds.py
-----------------------------------
It drew confidence intervals by bootstrapping over 10 KMeans seeds. For
the deterministic feature sets - NDVI, raw spectral, zero-shot Prithvi -
the features, the scaler and the PCA are identical on every seed, and
KMeans with n_init=10 converges to the same partition almost every time.
The ten "samples" were therefore near-duplicates, the interval collapsed
to near-zero width, and the benchmark appeared to separate models far more
sharply than the data supports. That is the degenerate-CI problem.

Seed variation measures only the clustering algorithm's own instability.
It says nothing about whether a result would hold on a different sample of
fields, which is the question a benchmark is for.

What this does instead
----------------------
Two sources of variation, reported separately:

1. SEED SPREAD - KMeans refitted with 10 seeds on all fields. Reported as
   IQM and min-max. A near-zero spread is stated plainly as algorithmic
   stability, not disguised as a confidence interval.

2. FIELD BOOTSTRAP - fields resampled with replacement; the whole pipeline
   (scale, PCA, cluster, score) is refitted on each resample. This gives a
   genuine interval, and it is non-degenerate for deterministic features
   because the data itself is being resampled.

3. PAIRED COMPARISON - every model is scored on the SAME resampled fields,
   so differences between models can be bootstrapped directly. The win
   rate and the interval on the difference are what support a claim that
   one representation beats another. An interval on each model separately
   does not: overlapping marginal intervals are compatible with a
   consistent paired difference.

Usage
-----
    python run_seeds_v2.py --year 2022
    python run_seeds_v2.py --n-boot 500
"""

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config


def iqm(values):
    """Interquartile mean - the GEO-Bench robust central estimate."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [25, 75])
    inner = values[(values >= lo) & (values <= hi)]
    return float(np.mean(inner)) if inner.size else float(np.mean(values))


def prepare(features, n_components=50, seed=0):
    """Standardise, then PCA if wide. Refitted inside every resample."""
    x = StandardScaler().fit_transform(features)
    if x.shape[1] > n_components:
        n = min(n_components, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=n, random_state=seed).fit_transform(x)
    return x


def score(features, seed, n_clusters):
    """Fit KMeans and return (silhouette, davies-bouldin)."""
    labels = KMeans(n_clusters=n_clusters, random_state=seed,
                    n_init=10).fit_predict(features)
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return (float(silhouette_score(features, labels)),
            float(davies_bouldin_score(features, labels)))


def load_models(year, data_dir, best_layer=5):
    """Assemble every feature set to be compared.

    `best_layer` is the probe-selected best layer from experiment A.
    Defaults to 5, which was the best trajectory layer in the 2022 run.
    Pass --best-layer to override.
    """
    targets_path = data_dir / f"field_targets_{year}.csv"
    if not targets_path.exists():
        print(f"{targets_path} missing - run compute_targets.py")
        sys.exit(1)
    targets = pd.read_csv(targets_path)

    emb_path = data_dir / f"prithvi_enc_{year}_none.pkl"
    if not emb_path.exists():
        print(f"{emb_path} missing - run extract_prithvi_v4.py")
        sys.exit(1)
    with open(emb_path, "rb") as fh:
        records = pickle.load(fh)

    by_id = {r["field_id"]: r for r in records}
    ndvi_cols = [c for c in ["ndvi_apr", "ndvi_jun", "ndvi_aug",
                             "ndre_apr", "ndre_jun", "ndre_aug",
                             "ndvi_greenup", "ndvi_senescence"]
                 if c in targets.columns]

    keep = targets["field_id"].isin(by_id) & targets[ndvi_cols].notna().all(axis=1)
    targets = targets[keep].reset_index(drop=True)
    ordered = [by_id[fid] for fid in targets["field_id"]]

    models = {
        "Prithvi frozen (final layer)":
            np.stack([r["embedding"] for r in ordered]).astype(np.float64),
        "NDVI/NDRE trajectory": targets[ndvi_cols].to_numpy(dtype=np.float64),
    }

    if "layers" in ordered[0]:
        layers = np.stack([r["layers"] for r in ordered]).astype(np.float64)
        n_layers = layers.shape[1]
        layer_idx = min(best_layer, n_layers - 1)
        models[f"Prithvi frozen (layer {layer_idx})"] = layers[:, layer_idx, :]

    spectral_path = data_dir / f"spectral_features_{year}.npy"
    if spectral_path.exists():
        spectral = np.load(spectral_path)
        if len(spectral) == len(keep):
            models["Raw spectral means"] = spectral[keep.to_numpy()].astype(
                np.float64)

    return models, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--n-seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--clusters", type=int, default=config.N_CLUSTERS)
    ap.add_argument("--best-layer", type=int, default=5,
                    help="probe-selected best layer from experiment A "
                         "(default 5, the 2022 result)")
    ap.add_argument("--data-dir", type=Path)
    args = ap.parse_args()

    data_dir = args.data_dir or config.DATA_DIR
    models, targets = load_models(args.year, data_dir,
                                  best_layer=args.best_layer)

    print("=" * 76)
    print(f"CLUSTERING BENCHMARK - {args.year}")
    print("=" * 76)
    print(f"Fields   : {len(targets)}")
    print(f"Clusters : {args.clusters}")
    print(f"Seeds    : {args.n_seeds}   Bootstrap resamples: {args.n_boot}")
    for name, feats in models.items():
        print(f"  {name:<34} {feats.shape}")

    # ---------------------------------------------------------------
    # 1. Seed spread on the full sample
    # ---------------------------------------------------------------
    print("\n" + "-" * 76)
    print("1. SEED SPREAD (algorithmic stability, all fields)")
    print("-" * 76)
    print(f"{'model':<34}{'sil IQM':>10}{'sil range':>20}{'DB IQM':>10}")

    prepared = {name: prepare(feats) for name, feats in models.items()}
    seed_rows = []
    for name, x in prepared.items():
        sils, dbs = [], []
        for seed in range(args.n_seeds):
            sil, db = score(x, seed, args.clusters)
            sils.append(sil)
            dbs.append(db)
            seed_rows.append({"model": name, "seed": seed,
                              "silhouette": sil, "davies_bouldin": db})
        spread = max(sils) - min(sils)
        print(f"{name:<34}{iqm(sils):>10.4f}"
              f"{f'[{min(sils):.4f}, {max(sils):.4f}]':>20}"
              f"{iqm(dbs):>10.4f}")
        if spread < 1e-6:
            print(f"{'':<34}  (identical across seeds - KMeans is stable "
                  f"here; this is not an interval)")

    # ---------------------------------------------------------------
    # 2. Paired field bootstrap
    # ---------------------------------------------------------------
    print("\n" + "-" * 76)
    print("2. FIELD BOOTSTRAP (sampling uncertainty, paired across models)")
    print("-" * 76)

    rng = np.random.default_rng(0)
    n_fields = len(targets)
    names = list(models)
    boot = {name: {"sil": [], "db": []} for name in names}

    for b in range(args.n_boot):
        idx = rng.choice(n_fields, n_fields, replace=True)
        for name in names:
            x = prepare(models[name][idx], seed=0)
            sil, db = score(x, 0, args.clusters)
            boot[name]["sil"].append(sil)
            boot[name]["db"].append(db)
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{args.n_boot} resamples")

    print(f"\n{'model':<34}{'sil':>9}{'95% CI':>20}{'DB':>9}{'95% CI':>20}")
    print("-" * 76)

    summary = []
    for name in names:
        sil = np.array(boot[name]["sil"], dtype=float)
        db = np.array(boot[name]["db"], dtype=float)
        sil_lo, sil_hi = np.nanpercentile(sil, [2.5, 97.5])
        db_lo, db_hi = np.nanpercentile(db, [2.5, 97.5])
        print(f"{name:<34}{np.nanmean(sil):>9.4f}"
              f"{f'[{sil_lo:.4f}, {sil_hi:.4f}]':>20}"
              f"{np.nanmean(db):>9.4f}"
              f"{f'[{db_lo:.4f}, {db_hi:.4f}]':>20}")
        summary.append({"model": name,
                        "silhouette_mean": float(np.nanmean(sil)),
                        "silhouette_lo": float(sil_lo),
                        "silhouette_hi": float(sil_hi),
                        "db_mean": float(np.nanmean(db)),
                        "db_lo": float(db_lo), "db_hi": float(db_hi)})

    # ---------------------------------------------------------------
    # 3. Paired differences
    # ---------------------------------------------------------------
    print("\n" + "-" * 76)
    print("3. PAIRED DIFFERENCES IN SILHOUETTE (same resampled fields)")
    print("-" * 76)
    print(f"{'comparison':<48}{'diff':>9}{'95% CI':>20}{'win':>7}")

    pairs = []
    for a, b in itertools.combinations(names, 2):
        diff = (np.array(boot[a]["sil"], dtype=float)
                - np.array(boot[b]["sil"], dtype=float))
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        win = float(np.mean(diff > 0))
        label = f"{a} - {b}"
        print(f"{label[:47]:<48}{np.nanmean(diff):>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}{win:>7.2f}")
        pairs.append({"model_a": a, "model_b": b,
                      "mean_diff": float(np.nanmean(diff)),
                      "lo": float(lo), "hi": float(hi), "win_rate": win})

    print("\nAn interval on the difference that excludes 0 is the claim you")
    print("can defend. Overlapping single-model intervals are not evidence")
    print("of no difference, which is why the paired form is reported.")

    out = data_dir / f"benchmark_v2_{args.year}"
    pd.DataFrame(seed_rows).to_csv(f"{out}_seeds.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{out}_bootstrap.csv", index=False)
    pd.DataFrame(pairs).to_csv(f"{out}_paired.csv", index=False)
    print(f"\nSaved: {out}_seeds.csv, _bootstrap.csv, _paired.csv")


if __name__ == "__main__":
    main()