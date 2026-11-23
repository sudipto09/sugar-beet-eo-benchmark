"""
Does the input-resolution / image-size choice change the conclusions?

Compares a baseline extraction (prithvi_enc_<year>_<mode>.pkl) with a
variant (…<mode><tag>.pkl) produced by
    python extract_prithvi_v4.py --mode none      --input-gsd 30 --tag _gsd30
    python extract_prithvi_v4.py --mode mask_last --input-gsd 30 --tag _gsd30
(or --img-size 224 --tag _s224 --batch-size 2).

Reports, for the same fields:
  * probe R^2 for the trajectory targets at the chosen layer and at the
    final layer, baseline vs variant, with the difference
  * linear CKA between the baseline and variant embeddings
  * Experiment C: paired bootstrap of (variant - baseline) R^2 for
    August NDVI from April + June
A change smaller than the ~0.03 noise floor means the gap does not matter
for these conclusions; a larger one means it does and must be reported.

Usage:
    python compare_variants.py --year 2022 --tag _gsd30 --layer 5
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
import probe_temporal as pt

NOISE_FLOOR = 0.03


def load(year, mode, tag, data_dir):
    path = data_dir / f"prithvi_enc_{year}_{mode}{tag}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()
    data_dir = args.data_dir or config.DATA_DIR
    config.DATA_DIR = data_dir

    targets = pd.read_csv(data_dir / f"field_targets_{args.year}.csv")
    base = load(args.year, "none", "", data_dir)
    var = load(args.year, "none", args.tag, data_dir)
    if base is None or var is None:
        raise SystemExit("need prithvi_enc_<year>_none.pkl and the tagged "
                         "variant; see the usage in the docstring")

    e_b, l_b, df_b = pt.align(base, targets)
    e_v, l_v, df_v = pt.align(var, targets)
    common = sorted(set(df_b.field_id) & set(df_v.field_id))
    ib = df_b.field_id.isin(common).to_numpy()
    iv = df_v.field_id.isin(common).to_numpy()
    df = df_b[ib].reset_index(drop=True)
    order_b = df_b[ib].field_id.tolist()
    order_v = df_v[iv].field_id.tolist()
    assert order_b == order_v, "field order differs between pickles"
    e_b, e_v = e_b[ib], e_v[iv]
    l_b, l_v = l_b[ib], l_v[iv]

    print("=" * 70)
    print(f"VARIANT '{args.tag}' vs baseline - {args.year}, "
          f"{len(df)} common fields, noise floor +/-{NOISE_FLOOR}")
    print("=" * 70)

    worst = 0.0
    for name, fb, fv in (
            (f"layer {args.layer}", l_b[:, args.layer, :], l_v[:, args.layer, :]),
            (f"final layer", e_b, e_v)):
        print(f"\n{name}   CKA(baseline, variant) = "
              f"{pt.linear_cka(fb, fv):.4f}")
        print(f"  {'target':<18}{'baseline':>10}{'variant':>10}{'diff':>9}")
        for t in pt.TARGETS:
            y = df[t].to_numpy()
            rb, rv = pt.probe_r2(fb, y), pt.probe_r2(fv, y)
            worst = max(worst, abs(rv - rb))
            print(f"  {t:<18}{rb:>10.4f}{rv:>10.4f}{rv - rb:>+9.4f}")

    mb = load(args.year, "mask_last", "", data_dir)
    mv = load(args.year, "mask_last", args.tag, data_dir)
    if mb is not None and mv is not None:
        xb, _, db = pt.align(mb, df)
        xv, _, dv = pt.align(mv, df)
        ids = sorted(set(db.field_id) & set(dv.field_id))
        kb = db.field_id.isin(ids).to_numpy()
        kv = dv.field_id.isin(ids).to_numpy()
        y = db[kb]["ndvi_aug"].to_numpy()
        d, lo, hi, pv, pb = pt.paired_bootstrap_diff(
            xv[kv], xb[kb], y, n_boot=args.n_boot)
        print(f"\nExp C (August NDVI from Apr+Jun, {len(ids)} fields)")
        print(f"  baseline R^2 {pb:.4f}   variant R^2 {pv:.4f}")
        print(f"  variant - baseline {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        worst = max(worst, abs(d))
    else:
        print("\n[skip] Exp C needs mask_last for both baseline and variant")

    print("\n" + "-" * 70)
    if worst <= NOISE_FLOOR:
        print(f"Largest change {worst:.3f} <= {NOISE_FLOOR}: the input "
              "variant does not change the conclusions.")
    else:
        print(f"Largest change {worst:.3f} > {NOISE_FLOOR}: the variant "
              "matters. Report both, and say which one the headline uses.")


if __name__ == "__main__":
    main()