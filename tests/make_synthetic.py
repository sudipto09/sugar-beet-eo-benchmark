"""
Build a synthetic stand-in for the extraction outputs.

Lets probe_temporal.py and run_seeds_v2.py be exercised end to end without
a GPU, the Prithvi checkpoint or the Sentinel-2 rasters, and - more
usefully - with a KNOWN answer, so the scripts can be checked for whether
they report the right thing rather than merely running.

Ground truth built in:
  * `none` embeddings encode the full three-date trajectory, so probes for
    green-up and senescence should score high.
  * `repeat_first` encodes April only, so those same probes MUST collapse
    toward zero. If they do not, the targets leak through a single date
    and experiment B's leakage check is what catches it.
  * `shuffle` is a near-rotation of `none`: high CKA, little R^2 change.
    This mimics an order-invariant encoder, the negative result the real
    experiment is designed to be able to detect.

Usage:
    python tests/make_synthetic.py --out /tmp/synth
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

N_FIELDS = 300
N_LAYERS = 12
DIM = 128
YEAR = 2022


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/synth"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    field_ids = [f"SB_FR_{i:04d}" for i in range(N_FIELDS)]

    # Latent phenology: emergence vigour, canopy peak, senescence rate.
    vigour = rng.normal(0, 1, N_FIELDS)
    peak = rng.normal(0, 1, N_FIELDS)
    senesce = rng.normal(0, 1, N_FIELDS)

    ndvi_apr = 0.20 + 0.05 * vigour + rng.normal(0, 0.01, N_FIELDS)
    ndvi_jun = 0.80 + 0.06 * peak + 0.02 * vigour + rng.normal(0, 0.01, N_FIELDS)
    ndvi_aug = 0.60 + 0.08 * senesce + 0.03 * peak + rng.normal(0, 0.01, N_FIELDS)

    targets = pd.DataFrame({
        "field_id": field_ids,
        "area_ha": rng.uniform(1, 20, N_FIELDS),
        "ndvi_apr": ndvi_apr,
        "ndvi_jun": ndvi_jun,
        "ndvi_aug": ndvi_aug,
        "ndre_apr": ndvi_apr * 0.6 + rng.normal(0, 0.01, N_FIELDS),
        "ndre_jun": ndvi_jun * 0.6 + rng.normal(0, 0.01, N_FIELDS),
        "ndre_aug": ndvi_aug * 0.6 + rng.normal(0, 0.01, N_FIELDS),
    })
    targets["ndvi_greenup"] = targets.ndvi_jun - targets.ndvi_apr
    targets["ndvi_senescence"] = targets.ndvi_aug - targets.ndvi_jun
    targets["ndvi_amplitude"] = (
        targets[["ndvi_apr", "ndvi_jun", "ndvi_aug"]].max(axis=1)
        - targets[["ndvi_apr", "ndvi_jun", "ndvi_aug"]].min(axis=1))
    targets.to_csv(args.out / f"field_targets_{YEAR}.csv", index=False)

    spectral = np.column_stack([
        ndvi_apr, ndvi_jun, ndvi_aug,
        rng.normal(0, 1, (N_FIELDS, 15)),
    ]).astype(np.float32)
    np.save(args.out / f"spectral_features_{YEAR}.npy", spectral)

    def embed(latents, seed, noise=0.25):
        """Project latents into a layered embedding stack."""
        gen = np.random.default_rng(seed)
        stack = np.zeros((N_FIELDS, N_LAYERS, DIM), dtype=np.float32)
        for layer in range(N_LAYERS):
            # Signal strengthens into the middle of the stack, then fades,
            # which is the depth profile layer-wise probing looks for.
            strength = 1.0 - abs(layer - N_LAYERS * 0.6) / N_LAYERS
            proj = gen.normal(0, 1, (latents.shape[1], DIM))
            stack[:, layer] = (strength * (latents @ proj)
                               + gen.normal(0, noise, (N_FIELDS, DIM)))
        return stack

    full = np.column_stack([vigour, peak, senesce])
    april_only = np.column_stack([vigour, np.zeros(N_FIELDS),
                                  np.zeros(N_FIELDS)])

    variants = {
        "none": embed(full, 1),
        "repeat_first": embed(april_only, 2),
        "shuffle": embed(full, 1) + np.random.default_rng(3).normal(
            0, 0.02, (N_FIELDS, N_LAYERS, DIM)),
        "reverse": embed(full[:, ::-1].copy(), 4),
        "mask_last": embed(np.column_stack([vigour, peak,
                                            np.zeros(N_FIELDS)]), 5),
    }

    for mode, stack in variants.items():
        records = [{
            "field_id": field_ids[i],
            "area_ha": float(targets.area_ha[i]),
            "embedding": stack[i, -1].astype(np.float32),
            "embedding_cls": stack[i, -1].astype(np.float32),
            "layers": stack[i].astype(np.float16),
            "date_order": [0, 1, 2],
        } for i in range(N_FIELDS)]

        path = args.out / f"prithvi_enc_{YEAR}_{mode}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(records, fh)
        print(f"  wrote {path.name}")

    print(f"\nSynthetic data in {args.out}")
    print("Expected when probing:")
    print("  none        -> green-up and senescence R^2 clearly positive")
    print("  repeat_first-> both collapse toward 0 (leakage check passes)")
    print("  shuffle     -> CKA near 1, little R^2 change")


if __name__ == "__main__":
    main()