"""
Does edge/mixed-pixel contamination explain the low field-mean NDVI?


"""

import argparse

import numpy as np

import config

EPS = 1e-8


def field_ndvi(src_red, src_nir, geom, offset, all_touched):
    from rasterio.mask import mask as rio_mask
    try:
        red, _ = rio_mask(src_red, [geom], crop=True, nodata=0,
                          all_touched=all_touched)
        nir, _ = rio_mask(src_nir, [geom], crop=True, nodata=0,
                          all_touched=all_touched)
    except Exception:                                      # noqa: BLE001
        return np.nan, 0
    red, nir = red[0].astype(np.float32), nir[0].astype(np.float32)
    ok = (red > 0) & (nir > 0)
    if not ok.any():
        return np.nan, 0
    r = (red[ok] - offset) / 10000.0
    n = (nir[ok] - offset) / 10000.0
    return float(np.mean((n - r) / (n + r + EPS))), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--date-index", type=int, default=1,
                    help="0=April 1=June 2=August")
    ap.add_argument("--erode", type=float, default=15.0,
                    help="metres to shrink polygons for the core estimate")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import geopandas as gpd
    import rasterio

    date = config.DATES[args.year][args.date_index]
    offset = config.boa_offset(date)
    gdf = gpd.read_file(config.fields_path(args.year))
    red_src = rasterio.open(config.SCENE_DIR / date / "B04.tif")
    nir_src = rasterio.open(config.SCENE_DIR / date / "B08.tif")
    gdf = gdf.to_crs(red_src.crs)
    if args.limit:
        gdf = gdf.head(args.limit)

    core_geom = gdf.geometry.buffer(-args.erode)

    rows = []
    for geom, core in zip(gdf.geometry, core_geom):
        t, nt = field_ndvi(red_src, nir_src, geom, offset, True)
        c, nc = field_ndvi(red_src, nir_src, geom, offset, False)
        k, nk = (np.nan, 0) if core.is_empty else \
            field_ndvi(red_src, nir_src, core, offset, False)
        rows.append((t, c, k, nt, nk))
    red_src.close()
    nir_src.close()

    a = np.array(rows, dtype=float)
    print("=" * 64)
    print(f"FIELD PURITY - {args.year} {date}  (offset {offset:.0f}, "
          f"erode {args.erode:.0f} m)")
    print("=" * 64)
    print(f"fields: {len(a)}   with core pixels: {np.isfinite(a[:, 2]).sum()}")
    for name, i in (("touched", 0), ("centre", 1), ("core", 2)):
        col = a[:, i][np.isfinite(a[:, i])]
        print(f"  {name:<8} mean {col.mean():.4f}  median {np.median(col):.4f}"
              f"  p95 {np.percentile(col, 95):.4f}")
    both = np.isfinite(a[:, 0]) & np.isfinite(a[:, 2])
    gain = (a[both, 2] - a[both, 0]).mean()
    print(f"\n  mean (core - touched) over {both.sum()} fields: {gain:+.4f}")
    print(f"  median pixels per field: touched {np.nanmedian(a[:, 3]):.0f}, "
          f"core {np.nanmedian(a[:, 4]):.0f}")
    if gain > 0.05:
        print("\n  Edge pixels dilute the field means. Switch "
              "compute_targets.py to core pixels and re-run the whole "
              "pipeline.")
    else:
        print("\n  Edge mixing does NOT explain the low NDVI. Look at crop "
              "label noise, stress, or radiometry next.")


if __name__ == "__main__":
    main()