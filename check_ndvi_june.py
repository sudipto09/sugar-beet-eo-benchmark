"""
Diagnostic: check whether June NDVI = 0.60 is canopy timing or a bad offset.

The sanity note in compute_targets.py expects June NDVI ~ 0.7-0.9 for
closed sugar beet canopy. The 2022 run showed a mean of 0.604, which is
below that range.  This script checks:

1. Percentile distribution of June NDVI (is the mean pulled down by a
   few bad fields, or is the whole distribution shifted?).
2. Comparison with the raw B08 and B04 reflectances that go into NDVI.
3. Whether the BOA offset was applied correctly (reversed check: compute
   NDVI with and without offset and compare).
4. Whether the 0.60 is plausible given the date (18 June) — sugar beet
   canopy closure in Franconia typically happens late June to mid-July,
   so 18 June may simply be pre-closure.

Usage:
    python check_ndvi_june.py --year 2022
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    targets_path = config.DATA_DIR / f"field_targets_{args.year}.csv"
    if not targets_path.exists():
        print(f"{targets_path} missing. Run compute_targets.py first.")
        sys.exit(1)

    df = pd.read_csv(targets_path)
    dates = config.DATES[args.year]

    print("=" * 68)
    print(f"JUNE NDVI DIAGNOSTIC - {args.year}")
    print("=" * 68)

    # --- 1. Percentile distribution ---
    print("\n1. NDVI distribution per date\n")
    for label, date in zip(config.DATE_LABELS, dates):
        col = f"ndvi_{label.lower()[:3]}"
        if col not in df.columns:
            print(f"  {label} ({date}): column '{col}' missing")
            continue
        series = df[col].dropna()
        pcts = np.percentile(series, [5, 10, 25, 50, 75, 90, 95])
        print(f"  {label} ({date}):  n={len(series)}")
        print(f"    mean={series.mean():.4f}  std={series.std():.4f}")
        print(f"    p5={pcts[0]:.4f}  p10={pcts[1]:.4f}  p25={pcts[2]:.4f}  "
              f"median={pcts[3]:.4f}")
        print(f"    p75={pcts[4]:.4f}  p90={pcts[5]:.4f}  p95={pcts[6]:.4f}")
        # Check for outliers
        low = (series < 0.0).sum()
        high = (series > 1.0).sum()
        if low or high:
            print(f"    WARNING: {low} fields <0, {high} fields >1 "
                  "(bad reflectance values)")

    # --- 2. Raw band reflectances ---
    print("\n2. Raw band reflectances (B04=Red, B08=NIR)\n")
    for label, date in zip(config.DATE_LABELS, dates):
        b04 = f"B04_{date}"
        b08 = f"B08_{date}"
        if b04 in df.columns and b08 in df.columns:
            red = df[b04].dropna()
            nir = df[b08].dropna()
            print(f"  {label} ({date}):")
            print(f"    B04 (Red): mean={red.mean():.4f}  "
                  f"median={red.median():.4f}  "
                  f"[p5={red.quantile(0.05):.4f}, p95={red.quantile(0.95):.4f}]")
            print(f"    B08 (NIR): mean={nir.mean():.4f}  "
                  f"median={nir.median():.4f}  "
                  f"[p5={nir.quantile(0.05):.4f}, p95={nir.quantile(0.95):.4f}]")
            # Manual NDVI from these
            ndvi_manual = (nir - red) / (nir + red + 1e-8)
            print(f"    NDVI from these: mean={ndvi_manual.mean():.4f}")
        else:
            print(f"  {label} ({date}): band columns not found in targets CSV")

    # --- 3. Offset check ---
    print("\n3. Offset verification\n")
    for date in dates:
        offset = config.boa_offset(date)
        print(f"  {date}: offset = {offset:.0f}")

    # --- 4. Phenological context ---
    print("\n4. Phenological context\n")
    print("  Typical Franconia sugar beet: emergence Apr-early May, canopy")
    print("  closure late Jun-mid Jul, senescence Sep-Oct. Reference values")
    print("  only; this script does not assume them in the verdict below.\n")
    for label, date in zip(config.DATE_LABELS, dates):
        day = f"{date[6:8]}/{date[4:6]}/{date[:4]}"
        col = f"ndvi_{label.lower()[:3]}"
        if col in df.columns:
            print(f"  {label} ({day}): mean NDVI = {df[col].mean():.3f}")

    # --- 5. Trajectory sanity ---
    print("\n5. Trajectory shape check\n")
    labels = [l.lower()[:3] for l in config.DATE_LABELS]
    cols = [f"ndvi_{l}" for l in labels]
    valid = df[cols].dropna()
    increasing_apr_jun = (valid[cols[1]] > valid[cols[0]]).mean()
    decreasing_jun_aug = (valid[cols[2]] < valid[cols[1]]).mean()
    peaked_at_jun = ((valid[cols[1]] > valid[cols[0]]) &
                     (valid[cols[1]] > valid[cols[2]])).mean()
    peaked_at_aug = (valid[cols[2]] > valid[cols[1]]).mean()

    print(f"  Fields where Jun > Apr:    {increasing_apr_jun:.1%}")
    print(f"  Fields where Aug < Jun:    {decreasing_jun_aug:.1%}")
    print(f"  Fields peaking at June:    {peaked_at_jun:.1%}")
    print(f"  Fields still rising (Aug > Jun): {peaked_at_aug:.1%}")
    print()
    if peaked_at_aug > 0.5:
        print("  Most fields are still rising into August, suggesting")
        print("  late canopy closure. June NDVI < 0.7 is expected.")
    elif peaked_at_jun > 0.5:
        print("  Most fields peak at June — classic early-closure pattern.")
    else:
        print("  Mixed trajectory shapes.")

    print("\n" + "=" * 68)
    print("VERDICT (computed from the data above)")
    m = df[cols].mean()
    jun_p95 = df[cols[1]].quantile(0.95)
    if m[cols[1]] < 0.7 and peaked_at_jun > 0.5 and m[cols[2]] < m[cols[1]]:
        print("  June is the seasonal peak (August is lower) yet the peak")
        print(f"  field-mean NDVI is only {m[cols[1]]:.2f} (p95 {jun_p95:.2f}).")
        print("  This is NOT explained by pre-closure canopy. Candidates:")
        print("   a) edge/mixed pixels diluting field means -> run")
        print("      check_field_purity.py")
        print("   b) summer stress (2022 was a hot, dry summer in Germany)")
        print("      lowering August; unverified here")
        print("   c) polygons including non-beet land (label noise)")
        print("   d) offset/scaling error: ruled out only if red is low")
        print("      (<0.06) over closed canopy; red here is "
              f"{df[f'B04_{dates[1]}'].mean():.3f}")
    elif m[cols[2]] > m[cols[1]]:
        print("  Still rising into August: June is pre-closure; low June")
        print("  NDVI is phenologically consistent.")
    else:
        print("  No clear pattern; inspect the percentiles manually.")
    print("=" * 68)


if __name__ == "__main__":
    main()