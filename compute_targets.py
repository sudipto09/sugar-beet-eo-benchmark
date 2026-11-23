"""
Per-field spectral targets and classical baselines, computed correctly.

Replaces the NDVI columns in benchmark_results_full.csv and the spectral
feature block that was computed inline inside run_10_seeds.py.

Two fixes over the earlier version:

1. NDVI used B07 as a "NIR proxy". B07 is a red-edge band at 783 nm; the
   standard NDVI uses the broad NIR at 842 nm (B08). Red-edge saturates
   differently over dense canopy, so the earlier trajectories compressed
   exactly where sugar beet canopy closure happens.

2. No radiometric offset was applied, so 2022 scenes (processing baseline
   >= 04.00) were biased by +1000 DN before the ratio was taken. A ratio
   index is not invariant to an additive offset: it pulls NDVI toward
   zero, and by a different amount at each reflectance level.

Outputs, per year:
    Data/field_targets_<year>.csv    field_id, ndvi_*, ndre_*, band means
    Data/spectral_features_<year>.npy   [n_fields, n_dates * n_bands]

Usage:
    python compute_targets.py --year 2022
"""

import argparse

import numpy as np
import pandas as pd

import config

EPS = 1e-8
SCALE = 10000.0          # Sentinel-2 L2A DN per unit reflectance


def field_band_means(gdf, dates, offsets, bands):
    """Mean surface reflectance per field, per date, per band."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from tqdm import tqdm

    # Open every raster once rather than per field.
    handles = {}
    for date in dates:
        for band in bands:
            path = config.SCENE_DIR / date / f"{band}.tif"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing - run download_prithvi_bands.py")
            handles[(date, band)] = rasterio.open(path)

    rows = []
    try:
        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  fields"):
            record = {"field_id": row["field_id"],
                      "area_ha": float(row.get("area_ha", 0.0))}

            for date in dates:
                for band in bands:
                    src = handles[(date, band)]
                    try:
                        out, _ = rio_mask(src, [row.geometry], crop=True,
                                          nodata=0, all_touched=True)
                        pixels = out[0].astype(np.float32)
                    except Exception:                      # noqa: BLE001
                        record[f"{band}_{date}"] = np.nan
                        continue

                    # 0 is nodata. Correct the offset, then scale.
                    valid = pixels[pixels > 0]
                    if valid.size == 0:
                        record[f"{band}_{date}"] = np.nan
                        continue

                    reflectance = (valid - offsets[date]) / SCALE
                    record[f"{band}_{date}"] = float(np.mean(reflectance))

            rows.append(record)
    finally:
        for src in handles.values():
            src.close()

    return pd.DataFrame(rows)


def add_indices(df, dates, labels):
    """NDVI and NDRE per date, from band means."""
    for date, label in zip(dates, labels):
        red = df[f"B04_{date}"]
        nir = df[f"B08_{date}"]          # broad NIR, the standard choice
        rededge = df[f"B05_{date}"]

        df[f"ndvi_{label}"] = (nir - red) / (nir + red + EPS)
        df[f"ndre_{label}"] = (nir - rededge) / (nir + rededge + EPS)

    # Trajectory shape features, which is what the temporal comparison
    # actually turns on.
    first, mid, last = labels
    df["ndvi_greenup"] = df[f"ndvi_{mid}"] - df[f"ndvi_{first}"]
    df["ndvi_senescence"] = df[f"ndvi_{last}"] - df[f"ndvi_{mid}"]
    df["ndvi_amplitude"] = (df[[f"ndvi_{l}" for l in labels]].max(axis=1)
                            - df[[f"ndvi_{l}" for l in labels]].min(axis=1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    config.describe()

    import geopandas as gpd
    import rasterio

    dates = config.DATES[args.year]
    labels = [l.lower()[:3] for l in config.DATE_LABELS]   # apr, jun, aug
    offsets = {d: config.boa_offset(d) for d in dates}

    print(f"Year {args.year}  dates {dates}")
    print(f"Offsets: {', '.join(f'{d}:{o:.0f}' for d, o in offsets.items())}\n")

    gdf = gpd.read_file(config.fields_path(args.year))
    with rasterio.open(config.SCENE_DIR / dates[0] / "B02.tif") as src:
        gdf = gdf.to_crs(src.crs)
    print(f"Fields: {len(gdf)}")

    df = field_band_means(gdf, dates, offsets, config.BASELINE_BANDS)
    df = add_indices(df, dates, labels)

    n_bad = df[[f"ndvi_{l}" for l in labels]].isna().any(axis=1).sum()
    if n_bad:
        print(f"\n[warn] {n_bad} fields have a missing date and will be "
              f"dropped downstream.")

    out_csv = config.DATA_DIR / f"field_targets_{args.year}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Spectral baseline: every band mean at every date, in a fixed order.
    cols = [f"{b}_{d}" for d in dates for b in config.BASELINE_BANDS]
    spectral = df[cols].to_numpy(dtype=np.float32)
    out_npy = config.DATA_DIR / f"spectral_features_{args.year}.npy"
    np.save(out_npy, spectral)
    print(f"Saved: {out_npy}  shape {spectral.shape}")

    print("\nNDVI summary (mean over fields):")
    for label in labels:
        series = df[f"ndvi_{label}"]
        print(f"  {label}: {series.mean():.3f}  "
              f"[{series.quantile(0.05):.3f}, {series.quantile(0.95):.3f}]")
    print("\nSanity: April should be low (bare soil, ~0.1-0.3), "
          "June high (closed canopy, ~0.7-0.9).")
    print("If April and June look similar, re-check the offset.")


if __name__ == "__main__":
    main()