"""
Temporal probing of frozen EO foundation model features.

What the frozen representation actually encodes about a field's season, and whether that information is
temporal or merely spectral.

Three experiments
-----------------
A. LAYER-WISE PROBING
   A ridge probe is fitted on every transformer block's pooled output and
   scored by cross-validated R^2 against agronomic targets. This locates
   the depth at which phenological information is most linearly
   accessible. Foundation models commonly peak in the middle of the stack
   and lose task-relevant detail in the last blocks, so reporting only the
   final layer - as the original pipeline did - can understate the model.

B. TEMPORAL FAITHFULNESS UNDER PERTURBATION
   The same fields are re-encoded with their dates reordered or removed.
   If a representation genuinely encodes temporal dynamics, permuting the
   date order must change it, and collapsing all three dates to one must
   destroy the trajectory information. If `shuffle` leaves the embedding
   almost unchanged, the model is order-invariant: it is pooling spectra
   over time rather than modelling a trajectory. Either outcome is a
   result, and this is the experiment that distinguishes them.

   Reported as cosine distance, linear CKA, and the drop in probe R^2.

C. HELD-OUT-DATE PREDICTION
   Features encoded from April and June only (mode `mask_last`) are probed
   for August NDVI. This tests extrapolation of phenology rather than
   reconstruction of inputs the model has already seen, so a trivially
   spectral representation cannot win it by memorising the target date.

Baselines throughout: NDVI/NDRE trajectory features and raw spectral band
means, which are the classical comparators the benchmark needs.

Usage
-----
    python probe_temporal.py                 # all available modes
    python probe_temporal.py --year 2022
    python probe_temporal.py --experiment A

Requires extract_prithvi_v4.py to have been run for at least mode `none`;
experiments B and C need the perturbed modes as well.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

ALPHAS = np.logspace(-2, 4, 13)
N_FOLDS = 5
TAG = ""     
TARGETS = {
    "ndvi_apr":        "April NDVI (single date)",
    "ndvi_jun":        "June NDVI (single date)",
    "ndvi_aug":        "August NDVI (single date)",
    "ndvi_greenup":    "Green-up (June - April)",
    "ndvi_senescence": "Senescence (August - June)",
    "ndvi_amplitude":  "Season amplitude",
}

TRAJECTORY_TARGETS = ["ndvi_greenup", "ndvi_senescence", "ndvi_amplitude"]




def load_mode(year, mode):
    """Load one extraction pickle, or None if it has not been produced."""
    path = config.DATA_DIR / f"prithvi_enc_{year}_{mode}{TAG}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        records = pickle.load(fh)
    return records


def align(records, targets_df):
    """Match embeddings to target rows on field_id, dropping incomplete rows."""
    by_id = {r["field_id"]: r for r in records}
    df = targets_df[targets_df["field_id"].isin(by_id)].copy()
    df = df.dropna(subset=list(TARGETS))

    ordered = [by_id[fid] for fid in df["field_id"]]
    embeddings = np.stack([r["embedding"] for r in ordered]).astype(np.float64)

    layers = None
    if "layers" in ordered[0]:
        layers = np.stack([r["layers"] for r in ordered]).astype(np.float64)

    return embeddings, layers, df.reset_index(drop=True)


# probing functions

def probe_r2(features, target, seed=0, groups=None):
    
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

    if groups is None:
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        predicted = cross_val_predict(model, features, target, cv=folds)
    else:
        folds = GroupKFold(n_splits=N_FOLDS)
        predicted = cross_val_predict(model, features, target, cv=folds,
                                      groups=groups)

    residual = np.sum((target - predicted) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return 1.0 - residual / total


def probe_r2_gb(features, target, seed=0):
    
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=seed))
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, features, target, cv=folds)
    residual = np.sum((target - predicted) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return 1.0 - residual / total


def probe_with_ci_gb(features, target, n_boot=1000, seed=0):
    """probe_with_ci but using gradient boosting."""
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=seed))
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, features, target, cv=folds)

    def r2(idx):
        truth, pred = target[idx], predicted[idx]
        total = np.sum((truth - truth.mean()) ** 2)
        if total < 1e-12:
            return np.nan
        return 1.0 - np.sum((truth - pred) ** 2) / total

    point = r2(np.arange(len(target)))
    rng = np.random.default_rng(seed)
    n = len(target)
    scores = [r2(rng.choice(n, n, replace=True)) for _ in range(n_boot)]
    scores = [s for s in scores if np.isfinite(s)]
    lo, hi = np.percentile(scores, [2.5, 97.5]) if scores else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def is_construction_ceiling(name, target):
    
    return (name in ("ndvi_ndre", "spectral")
            and target in ("ndvi_greenup", "ndvi_senescence"))


def paired_bootstrap_diff(features_a, features_b, target, n_boot=1000, seed=0):
    
    model_a = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model_b = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    pred_a = cross_val_predict(model_a, features_a, target, cv=folds)
    pred_b = cross_val_predict(model_b, features_b, target, cv=folds)

    def r2_pair(idx):
        truth = target[idx]
        total = np.sum((truth - truth.mean()) ** 2)
        if total < 1e-12:
            return np.nan, np.nan
        r2_a = 1.0 - np.sum((truth - pred_a[idx]) ** 2) / total
        r2_b = 1.0 - np.sum((truth - pred_b[idx]) ** 2) / total
        return r2_a - r2_b, (r2_a, r2_b)

    all_idx = np.arange(len(target))
    point_diff, (point_a, point_b) = r2_pair(all_idx)

    rng = np.random.default_rng(seed)
    n = len(target)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        d, _ = r2_pair(idx)
        if np.isfinite(d):
            diffs.append(d)

    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return float(point_diff), float(lo), float(hi), float(point_a), float(point_b)


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=0):
    """Percentile CI of the mean, resampling the entries."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, values.size, replace=True).mean()
             for _ in range(n_boot)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def probe_with_ci(features, target, n_boot=1000, seed=0):
    
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, features, target, cv=folds)

    def r2(idx):
        truth, pred = target[idx], predicted[idx]
        total = np.sum((truth - truth.mean()) ** 2)
        if total < 1e-12:
            return np.nan
        return 1.0 - np.sum((truth - pred) ** 2) / total

    point = r2(np.arange(len(target)))

    rng = np.random.default_rng(seed)
    n = len(target)
    scores = [r2(rng.choice(n, n, replace=True)) for _ in range(n_boot)]
    scores = [s for s in scores if np.isfinite(s)]

    lo, hi = np.percentile(scores, [2.5, 97.5]) if scores else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


#representation similarity metrics

def linear_cka(x, y):
    """Linear CKA between two representation matrices of the same rows.

    1.0 means the two representations are identical up to rotation and
    scale; lower means the perturbation genuinely changed the geometry.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    if xx * yy == 0:
        return np.nan
    return float(cross / (xx * yy))


def mean_cosine_distance(x, y):
    """Mean cosine distance between matched rows of two embeddings."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(1.0 - np.sum(xn * yn, axis=1)))



# Experiments


def experiment_a(layers, embeddings, baselines, df, n_boot):
    """Layer-wise probing."""
    print("\n" + "=" * 74)
    print("A. LAYER-WISE PROBING - cross-validated R^2")
    print("=" * 74)

    rows = []

    if layers is None:
        print("\nper-layer output not in the pickle. Re-run "
              "extract_prithvi_v4.py without --no-layers.")
    else:
        n_layers = layers.shape[1]
        print(f"\nPrithvi encoder, {n_layers} blocks\n")
        header = "layer  " + "  ".join(f"{t:>16}" for t in TARGETS)
        print(header)
        print("-" * len(header))

        for layer_idx in range(n_layers):
            feats = layers[:, layer_idx, :]
            scores = {t: probe_r2(feats, df[t].to_numpy()) for t in TARGETS}
            rows.append({"features": f"prithvi_layer_{layer_idx:02d}",
                         "layer": layer_idx, **scores})
            print(f"{layer_idx:>5}  " +
                  "  ".join(f"{scores[t]:>16.4f}" for t in TARGETS))

        traj = np.array([[r[t] for t in TRAJECTORY_TARGETS] for r in rows])
        best = int(np.argmax(traj.mean(axis=1)))
        print(f"\nBest layer for trajectory targets: {best} "
              f"(mean R^2 {traj[best].mean():.4f})")
        print(f"Final layer:                       {len(rows) - 1} "
              f"(mean R^2 {traj[-1].mean():.4f})")
        if traj[best].mean() > traj[-1].mean() + 0.01:
            print("The final layer is NOT the best probe point - report the "
                  "layer-wise curve, not just the last block.")

    print("\nBaselines and final-layer Prithvi, with 95% CI over fields:\n")
    named = {"prithvi_final": embeddings, **baselines}
    any_ceiling = False
    for name, feats in named.items():
        for target in TRAJECTORY_TARGETS:
            point, lo, hi = probe_with_ci(feats, df[target].to_numpy(),
                                          n_boot=n_boot)
            ceiling = is_construction_ceiling(name, target)
            any_ceiling |= ceiling
            flag = "  <- ceiling, not a rival" if ceiling else ""
            print(f"  {name:<18} {target:<18} "
                  f"R^2 {point:>7.4f}  [{lo:>7.4f}, {hi:>7.4f}]{flag}")
            rows.append({"features": name, "layer": np.nan,
                         "target": target, "r2": point,
                         "lo": lo, "hi": hi,
                         "construction_ceiling": ceiling})

    if any_ceiling:
        print("\n  Green-up is ndvi_jun - ndvi_apr and senescence is")
        print("  ndvi_aug - ndvi_jun. Both are linear functions of the")
        print("  NDVI/NDRE baseline's own columns, so that baseline scores")
        print("  R^2 = 1 by construction. Read it as the ceiling the probe")
        print("  could reach, and do NOT report it as a model Prithvi lost")
        print("  to. Experiment C is the comparison that is actually fair,")
        print("  because there the target is withheld from both sides.")

    return pd.DataFrame(rows)


def experiment_b(year, reference, df, baselines, n_boot, best_layer_ref=None,
                 best_layer_idx=None):
    """Temporal faithfulness under perturbation."""
    print("\n" + "=" * 74)
    print("B. TEMPORAL FAITHFULNESS UNDER PERTURBATION")
    print("=" * 74)

    modes = ["shuffle", "reverse", "repeat_first"]
    available = {}
    for mode in modes:
        records = load_mode(year, mode)
        if records is None:
            print(f"\n mode '{mode}' not extracted. Run: "
                  f"python extract_prithvi_v4.py --mode {mode}")
            continue
        available[mode] = records

    if not available:
        return pd.DataFrame()

    ref_r2 = {t: probe_r2(reference, df[t].to_numpy())
              for t in TRAJECTORY_TARGETS}

    print("\nReference (mode 'none', final layer) trajectory R^2:")
    for target, value in ref_r2.items():
        print(f"  {target:<18} {value:>7.4f}")

    if best_layer_ref is not None:
        best_r2 = {t: probe_r2(best_layer_ref, df[t].to_numpy())
                   for t in TRAJECTORY_TARGETS}
        print(f"\nReference (mode 'none', layer {best_layer_idx}) trajectory R^2:")
        for target, value in best_r2.items():
            print(f"  {target:<18} {value:>7.4f}")

  
    apr_idx_cols = [c for c in ["ndvi_apr", "ndre_apr"] if c in df.columns]
    if apr_idx_cols:
        apr_idx_features = df[apr_idx_cols].to_numpy()
        apr_idx_r2 = {t: probe_r2(apr_idx_features, df[t].to_numpy())
                      for t in TRAJECTORY_TARGETS}
        print("\nApril-only NDVI/NDRE baseline (leakage reference):")
        for target, value in apr_idx_r2.items():
            print(f"  {target:<18} {value:>7.4f}")

    # April-only raw spectral bands (6 bands) 
    spectral_path = config.DATA_DIR / f"spectral_features_{year}.npy"
    apr_raw_r2 = None
    if spectral_path.exists():
        spectral_all = np.load(spectral_path)
        targets_path = config.DATA_DIR / f"field_targets_{year}.csv"
        targets_full = pd.read_csv(targets_path)
        keep = targets_full["field_id"].isin(df["field_id"]).to_numpy()
        if len(spectral_all) == len(targets_full):
            spectral_kept = spectral_all[keep]
            # spectral: 18 cols = 6 bands × 3 dates, date-major order.
            # April = cols 0:6.
            apr_raw = spectral_kept[:, :6]
            apr_raw_r2 = {t: probe_r2(apr_raw, df[t].to_numpy())
                          for t in TRAJECTORY_TARGETS}
            print("\nApril-only raw spectral bands (6 bands, leakage reference):")
            for target, value in apr_raw_r2.items():
                print(f"  {target:<18} {value:>7.4f}")

    print("  (repeat_first R^2 should be near these, not near zero)")

    print("\n" + "-" * 74)
    print(f"{'perturbation':<16}{'cos dist':>10}{'CKA':>8}"
          f"{'greenup':>10}{'senesc.':>10}{'amplitude':>11}")
    print("-" * 74)

    rows = []
    paired_rows = []
    for mode, records in available.items():
        perturbed, p_layers, pdf = align(records, df)

        
        shared = pdf["field_id"].isin(df["field_id"])
        shared_np = shared.to_numpy()
        perturbed = perturbed[shared_np]
        common = pdf.loc[shared].reset_index(drop=True)
        ref_idx = df["field_id"].isin(common["field_id"]).to_numpy()
        ref_subset = reference[ref_idx]

        cos = mean_cosine_distance(ref_subset, perturbed)
        cka = linear_cka(ref_subset, perturbed)

        scores = {t: probe_r2(perturbed, common[t].to_numpy())
                  for t in TRAJECTORY_TARGETS}

        print(f"{mode:<16}{cos:>10.4f}{cka:>8.4f}"
              f"{scores['ndvi_greenup']:>10.4f}"
              f"{scores['ndvi_senescence']:>10.4f}"
              f"{scores['ndvi_amplitude']:>11.4f}")

        row = {"perturbation": mode, "cosine_distance": cos, "cka": cka,
               **{f"r2_{k}": v for k, v in scores.items()},
               **{f"delta_r2_{k}": scores[k] - ref_r2[k] for k in scores}}

        
        variants = [("final", perturbed, ref_subset)]
        if (best_layer_ref is not None and p_layers is not None
                and best_layer_idx is not None):
            li = min(best_layer_idx, p_layers.shape[1] - 1)
            variants.append((f"layer{li}", p_layers[shared_np][:, li, :],
                             best_layer_ref[ref_idx]))
        for label, feat_p, feat_r in variants:
            for t in TRAJECTORY_TARGETS:
                d, lo, hi, _, _ = paired_bootstrap_diff(
                    feat_p, feat_r, common[t].to_numpy(), n_boot=n_boot)
                row[f"pdiff_{label}_{t}"] = d
                row[f"pdiff_{label}_{t}_lo"] = lo
                row[f"pdiff_{label}_{t}_hi"] = hi
                paired_rows.append((mode, label, t, d, lo, hi))
        rows.append(row)

    print("-" * 74)

    print("\nPaired R^2 change (perturbed - unperturbed), 95% bootstrap CI")
    print("  '*' = interval excludes zero (change distinguishable from noise)")
    print(f"  {'perturbation':<14}{'layer':<9}{'target':<17}"
          f"{'change':>8}  {'95% CI':<20}")
    for mode, label, t, d, lo, hi in paired_rows:
        flag = "*" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else " "
        print(f"  {mode:<14}{label:<9}{t:<17}{d:>+8.3f}  "
              f"[{lo:>+7.3f}, {hi:>+7.3f}] {flag}")
    print("  Note: shuffle/reverse CIs that include 0 support order-invariance;")
    print("  a '*' with a small change is detectable but may be practically minor.")

    print("\nHow to read this:")
    print("  shuffle / reverse with CKA near 1.00 and no R^2 drop")
    print("      -> the encoder is order-invariant: it pools spectra over")
    print("         time rather than representing a trajectory.")
    print("  repeat_first R^2 close to April-only baseline above")
    print("      -> expected: the residual R^2 comes from April state")
    print("         correlating with the trajectory, not from leakage.")
    print("  repeat_first R^2 much HIGHER than April-only baseline")
    print("      -> the target leaks through a single date; fix the target")
    print("         before drawing any conclusion about the model.")

    return pd.DataFrame(rows)


def experiment_c(year, df, baselines, n_boot):
    """Held-out-date prediction."""
    print("\n" + "=" * 74)
    print("C. HELD-OUT-DATE PREDICTION - August NDVI from April + June")
    print("=" * 74)

    records = load_mode(year, "mask_last")
    if records is None:
        print("\n mode 'mask_last' not extracted. Run: "
              "python extract_prithvi_v4.py --mode mask_last")
        return pd.DataFrame()

    masked, _, mdf = align(records, df)
    target = mdf["ndvi_aug"].to_numpy()

    # Baseline with the same information: April and June NDVI/NDRE only.
    early_cols = ["ndvi_apr", "ndvi_jun", "ndre_apr", "ndre_jun"]
    early_cols = [c for c in early_cols if c in mdf.columns]
    early = mdf[early_cols].to_numpy()

    
    spectral_path = config.DATA_DIR / f"spectral_features_{year}.npy"
    spectral_early = None
    if spectral_path.exists():
        spectral_all = np.load(spectral_path)
        targets_path = config.DATA_DIR / f"field_targets_{year}.csv"
        targets_full = pd.read_csv(targets_path)
        # Keep same rows as mdf
        keep = targets_full["field_id"].isin(mdf["field_id"]).to_numpy()
        if len(spectral_all) == len(targets_full):
            spectral_kept = spectral_all[keep]
            # 18 cols = 6 bands × 3 dates. compute_targets writes columns
            # in DATE_LABELS order: April (0:6), June (6:12), August (12:18).
            spectral_early = spectral_kept[:, :12]  # Apr + Jun only

    print(f"\nFields: {len(target)}   predicting August NDVI\n")

    rows = []

    # ridge probe on the frozen representation, and on the baselines with the same information.
    print("  Linear ridge probes:")
    contenders = {"prithvi (Apr+Jun encoded)": masked,
                  "NDVI/NDRE (Apr+Jun)": early}
    if spectral_early is not None:
        contenders["spectral (Apr+Jun, 12 feat)"] = spectral_early

    for name, feats in contenders.items():
        point, lo, hi = probe_with_ci(feats, target, n_boot=n_boot)
        print(f"    {name:<34} R^2 {point:>7.4f}  [{lo:>7.4f}, {hi:>7.4f}]")
        rows.append({"features": name, "model": "ridge",
                     "r2": point, "lo": lo, "hi": hi})

    # Gradient-boosted trees 
    print("\n  Gradient boosting (capacity control):")
    gb_contenders = {"NDVI/NDRE (Apr+Jun) GB": early}
    if spectral_early is not None:
        gb_contenders["spectral (Apr+Jun) GB"] = spectral_early

    for name, feats in gb_contenders.items():
        point, lo, hi = probe_with_ci_gb(feats, target, n_boot=n_boot)
        print(f"    {name:<34} R^2 {point:>7.4f}  [{lo:>7.4f}, {hi:>7.4f}]")
        rows.append({"features": name, "model": "gb",
                     "r2": point, "lo": lo, "hi": hi})

    print("\n  If GB on 12 spectral features matches Prithvi ridge on 1024,")
    print("  the advantage was model capacity, not richer encoding.")
    print("  If Prithvi still leads, the frozen features carry information")
    print("  the raw bands do not, even with a flexible learner.")

    # --- Paired bootstrap: Prithvi vs raw spectral (Apr+Jun) ---
    if spectral_early is not None:
        print("\n  Paired bootstrap (same fields, same folds):")
        diff, lo, hi, pa, pb = paired_bootstrap_diff(
            masked, spectral_early, target, n_boot=n_boot)
        print(f"    Prithvi ridge R^2:             {pa:>7.4f}")
        print(f"    Spectral (12 feat) ridge R^2:  {pb:>7.4f}")
        print(f"    Difference (Prithvi - spectral): {diff:>+7.4f}  "
              f"[{lo:>+7.4f}, {hi:>+7.4f}]")
        if lo <= 0 <= hi:
            print("    -> interval contains zero: no significant advantage.")
        elif lo > 0:
            print("    -> Prithvi significantly better than raw spectral.")
        else:
            print("    -> raw spectral significantly better than Prithvi.")
        rows.append({"features": "paired_diff (prithvi - spectral)",
                     "model": "paired_bootstrap",
                     "r2": diff, "lo": lo, "hi": hi})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--experiment", choices=["A", "B", "C"], action="append",
                    help="run only these (default: all)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--best-layer", type=int, default=None,
                    help="probe-selected best layer (from experiment A). "
                         "If given, experiment B also reports this layer.")
    ap.add_argument("--data-dir", type=Path,
                    help="override config.DATA_DIR (used by the tests)")
    ap.add_argument("--tag", default="",
                    help="read prithvi_enc_<year>_<mode><tag>.pkl, e.g. "
                         "_gsd30 from extract_prithvi_v4.py --tag _gsd30")
    ap.add_argument("--min-area-ha", type=float, default=0.0,
                    help="sensitivity analysis: drop fields smaller than "
                         "this (ha). Area is independent of the NDVI "
                         "targets, so this filter is not circular.")
    args = ap.parse_args()
    global TAG
    TAG = args.tag

    if args.data_dir:
        config.DATA_DIR = args.data_dir

    wanted = args.experiment or ["A", "B", "C"]

    targets_path = config.DATA_DIR / f"field_targets_{args.year}.csv"
    if not targets_path.exists():
        print(f"{targets_path} missing - run: "
              f"python compute_targets.py --year {args.year}")
        sys.exit(1)
    targets_df = pd.read_csv(targets_path)

    records = load_mode(args.year, "none")
    if records is None:
        print(f"prithvi_enc_{args.year}_none.pkl missing - run: "
              f"python extract_prithvi_v4.py --year {args.year}")
        sys.exit(1)

    embeddings, layers, df = align(records, targets_df)

    if args.min_area_ha > 0:
        keep_rows = (df["area_ha"] >= args.min_area_ha).to_numpy()
        print(f"[sensitivity] --min-area-ha {args.min_area_ha}: keeping "
              f"{keep_rows.sum()} of {len(df)} fields")
        embeddings = embeddings[keep_rows]
        layers = layers[keep_rows] if layers is not None else None
        df = df[keep_rows].reset_index(drop=True)

    print("=" * 74)
    print(f"TEMPORAL PROBING - {args.year}")
    print("=" * 74)
    print(f"Fields matched : {len(df)}")
    print(f"Embedding dim  : {embeddings.shape[1]}")
    print(f"Layers stored  : {layers.shape[1] if layers is not None else 0}")

    if embeddings.shape[1] == 1536:
        print("\n1536-dim features are decoder reconstructions, not")
        print("encoder features. Re-extract with extract_prithvi_v4.py.")
        sys.exit(1)

    # Classical baselines on the same fields.
    ndvi_cols = [c for c in
                 ["ndvi_apr", "ndvi_jun", "ndvi_aug", "ndre_apr",
                  "ndre_jun", "ndre_aug"] if c in df.columns]
    baselines = {"ndvi_ndre": df[ndvi_cols].to_numpy()}

    spectral_path = config.DATA_DIR / f"spectral_features_{args.year}.npy"
    if spectral_path.exists():
        spectral = np.load(spectral_path)
        if len(spectral) == len(targets_df):
            keep = targets_df["field_id"].isin(df["field_id"]).to_numpy()
            baselines["spectral"] = spectral[keep]

    # Best layer features (for experiment B reporting)
    best_layer_ref = None
    best_layer_idx = args.best_layer
    if best_layer_idx is not None and layers is not None:
        idx = min(best_layer_idx, layers.shape[1] - 1)
        best_layer_ref = layers[:, idx, :]
        print(f"Best layer    : {idx} (from --best-layer)")

    outputs = {}
    if "A" in wanted:
        outputs["layerwise"] = experiment_a(layers, embeddings, baselines,
                                            df, args.n_boot)
    if "B" in wanted:
        outputs["perturbation"] = experiment_b(args.year, embeddings, df,
                                               baselines, args.n_boot,
                                               best_layer_ref=best_layer_ref,
                                               best_layer_idx=best_layer_idx)
    if "C" in wanted:
        outputs["heldout"] = experiment_c(args.year, df, baselines,
                                          args.n_boot)

    print("\n" + "=" * 74)
    for name, frame in outputs.items():
        if frame is None or frame.empty:
            continue
        sens = f"_minarea{args.min_area_ha:g}" if args.min_area_ha > 0 else ""
        out = config.DATA_DIR / f"probe_{name}_{args.year}{TAG}{sens}.csv"
        frame.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()