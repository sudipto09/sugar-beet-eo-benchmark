"""
Do the sugar beet polygons peak after June? 

"""
import argparse

import numpy as np
import pandas as pd

import config
import compute_targets as ct

EPS = 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD of the extra scene")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    if args.download:
        import download_prithvi_bands as dl
        ok = dl.fetch_scene(dl.open_catalog(), args.date)
        if not ok:
            raise SystemExit("download incomplete; pick another date")

    import geopandas as gpd
    import rasterio

    gdf = gpd.read_file(config.fields_path(args.year))
    with rasterio.open(config.SCENE_DIR / args.date / "B04.tif") as src:
        gdf = gdf.to_crs(src.crs)

    offsets = {args.date: config.boa_offset(args.date)}
    print(f"Extra date {args.date}, offset {offsets[args.date]:.0f} "
          "(unverified if the warning above appeared; run "
          "check_processing_baseline.py for it)")
    extra = ct.field_band_means(gdf, [args.date], offsets, ["B04", "B08"])
    red, nir = extra[f"B04_{args.date}"], extra[f"B08_{args.date}"]
    extra["ndvi_extra"] = (nir - red) / (nir + red + EPS)

    base = pd.read_csv(config.DATA_DIR / f"field_targets_{args.year}.csv")
    df = base.merge(extra[["field_id", "ndvi_extra"]], on="field_id")
    df = df.dropna(subset=["ndvi_apr", "ndvi_jun", "ndvi_aug", "ndvi_extra"])

    dates = config.DATES[args.year] + [args.date]
    series = {d: v for d, v in zip(
        dates, [df.ndvi_apr, df.ndvi_jun, df.ndvi_aug, df.ndvi_extra])}
    order = sorted(series)                     # chronological
    cols = {config.DATES[args.year][0]: "ndvi_apr",
            config.DATES[args.year][1]: "ndvi_jun",
            config.DATES[args.year][2]: "ndvi_aug", args.date: "ndvi_extra"}

    print(f"\n{len(df)} fields with all four dates\n")
    print(f"{'date':<10}{'mean':>8}{'p5':>8}{'median':>8}{'p95':>8}")
    for d in order:
        v = df[cols[d]]
        print(f"{d:<10}{v.mean():>8.3f}{v.quantile(.05):>8.3f}"
              f"{v.median():>8.3f}{v.quantile(.95):>8.3f}")

    mat = np.stack([df[cols[d]].to_numpy() for d in order], axis=1)
    peak = np.argmax(mat, axis=1)
    print("\nWhich date is each field's maximum?")
    for i, d in enumerate(order):
        print(f"  {d}: {np.mean(peak == i):.1%}")

    print("\nRead: if the July date is the most common peak and its mean is "
          "clearly above June, the low August value is a real late-season "
          "drop. If July is not above June, suspect the polygons.")


if __name__ == "__main__":
    main()