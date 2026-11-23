"""
Cut sugar beet field polygons for ONE YEAR from the DLR CropTypes map.

Same method as extract_fields_highres.py (vectorise the sugar beet pixels,
keep 0.5-100 ha), but:
  * it reads the downloadable v21 GeoTIFF, where sugar beet is class 60
    (NOT 13, which is winter rye in that legend);
  * it works for any year, so 2021 gets its own correct labels instead of
    reusing the 2022 polygons.

The 2022 polygons already in use came from the WMS layer
CROPTYPES_DE_P1Y_V02 and were verified against the V021 map (98.1% have
sugar beet as majority class), so they are NOT regenerated. 2021 therefore
comes from V021 and 2022 from V02: two product versions. Pass --year 2022
with --out elsewhere if you want a like-for-like 2022 set from V021, but
that would change the 2022 field set and require re-extracting 2022.

Use --like to restrict to the extent of the existing 2022 polygons, so both
years cover the same area:

    python make_polygons_from_croptypes.py --year 2021 \\
        --raster https://download.geoservice.dlr.de/CROPTYPES/files/CROPTYPES_DE_P1Y_2021_V021/CROPTYPES_DE_P1Y_2021_V021.tif \\
        --like Data/sugar_beet_fields_franconia_hires.geojson

Output: Data/sugar_beet_fields_<year>.geojson (EPSG:32632), with field_id,
area_ha, crop_class, crop_name, year.
"""

import argparse
from pathlib import Path

import numpy as np

import config

SUGAR_BEET_CLASS = 60
MIN_HA, MAX_HA = 0.5, 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--raster", required=True, help="path or URL of the "
                    "DLR CropTypes V021 GeoTIFF for that year")
    ap.add_argument("--like", type=Path,
                    help="geojson whose extent to reuse (default: "
                         "config.BBOX)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import shape

    out_path = args.out or config.DATA_DIR / f"sugar_beet_fields_{args.year}.geojson"

    with rasterio.open(args.raster) as ras:
        if args.like:
            ref = gpd.read_file(args.like).to_crs(ras.crs)
            bounds = tuple(float(b) for b in ref.total_bounds)
            print(f"Extent from {args.like.name}: {bounds}")
        else:
            bounds = transform_bounds("EPSG:4326", ras.crs, *config.BBOX)
            print(f"Extent from config.BBOX: {bounds}")

        win = from_bounds(*bounds, ras.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(Window(0, 0, ras.width, ras.height))
        except Exception:                                  # noqa: BLE001
            raise SystemExit("Extent does not overlap this raster.")
        data = ras.read(1, window=win)
        transform = ras.window_transform(win)
        crs = ras.crs

    present = sorted(int(c) for c in np.unique(data) if c)
    print(f"Window {data.shape}; classes present: {present}")
    if SUGAR_BEET_CLASS not in present:
        raise SystemExit(f"Class {SUGAR_BEET_CLASS} (sugar beet) not in this "
                         "window - wrong raster or year?")

    mask = data == SUGAR_BEET_CLASS
    n_pix = int(mask.sum())
    print(f"Sugar beet pixels: {n_pix} ({100 * n_pix / data.size:.2f}%)")

    polygons = [shape(geom) for geom, val in
                shapes(mask.astype(np.uint8), mask=mask, transform=transform)
                if val == 1]
    print(f"Raw patches: {len(polygons)}")

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf["area_ha"] = gdf.to_crs("EPSG:32632").geometry.area / 10000.0
    print("Area before filter: "
          f"<{MIN_HA} ha {int((gdf.area_ha < MIN_HA).sum())}, "
          f"{MIN_HA}-{MAX_HA} ha "
          f"{int(((gdf.area_ha >= MIN_HA) & (gdf.area_ha <= MAX_HA)).sum())}, "
          f">{MAX_HA} ha {int((gdf.area_ha > MAX_HA).sum())}")

    gdf = gdf[(gdf.area_ha >= MIN_HA) & (gdf.area_ha <= MAX_HA)].copy()
    gdf = gdf.reset_index(drop=True)
    prefix = f"SB{str(args.year)[2:]}"
    gdf["field_id"] = [f"{prefix}_{i:04d}" for i in range(len(gdf))]
    gdf["crop_class"] = SUGAR_BEET_CLASS
    gdf["crop_name"] = "sugar beet"
    gdf["year"] = args.year
    gdf = gdf.to_crs("EPSG:32632")[["field_id", "area_ha", "crop_class",
                                     "crop_name", "year", "geometry"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"\nKept {len(gdf)} fields: area min {gdf.area_ha.min():.2f}, "
          f"median {gdf.area_ha.median():.2f}, mean {gdf.area_ha.mean():.2f}, "
          f"max {gdf.area_ha.max():.1f} ha")
    print(f"Saved: {out_path}")
    print("\nCompare the count with the 2022 set (1020). A very different "
          "number means the extent or the product version differs.")


if __name__ == "__main__":
    main()