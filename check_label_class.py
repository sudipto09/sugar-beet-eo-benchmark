"""
What crop are the 1020 polygons, really?

The polygons were cut from the DLR WMS layer CROPTYPES_DE_P1Y_V02 by
selecting raster value 13 (extract_fields_highres.py). The downloadable
DLR product has a legend (LEGEND_CropTypes_v21.txt) in which code 13 is
WINTER RYE and sugar beet is code 60. The WMS layer appears to serve
ordinal class indices instead (13 = the 13th class = sugar beet), but that
was inferred, not verified, and the legend saved by map_all_classes.py has
duplicate and missing entries. This script settles it from two independent
angles, using the downloadable v21 raster for 2022:

 1. COMPOSITION: which v21 class lies under each polygon. If the polygons
    are sugar beet, class 60 should dominate.
 2. FINGERPRINT: field-mean NDVI at the three dates for random pixels of
    each major v21 class, compared with the polygons' mean trajectory.
    This does not depend on any label code: it asks which crop the
    polygons behave like.

Get the raster: open download.geoservice.dlr.de/CROPTYPES/files/ , take the
2022 GeoTIFF (10 m, EPSG:32632, all of Germany, Cloud Optimized, so only
the window around your fields is read) and pass its path or URL:

    python check_label_class.py --raster <path-or-url-to-2022-croptypes.tif>
"""

import argparse

import numpy as np
import pandas as pd

import config

V21 = {11: "winter wheat", 12: "winter barley", 13: "winter rye",
       14: "other winter cereals", 21: "spring wheat", 22: "spring barley",
       23: "spring oat", 30: "maize", 40: "legumes", 41: "soybeans",
       42: "lupins", 50: "potatoes", 60: "sugar beet", 71: "rapeseed",
       72: "sunflowers", 83: "permanent grassland", 84: "forage",
       90: "vineyard", 100: "woody vegetation", 120: "vegetables",
       110: "hops", 111: "other agricultural use"}

FINGERPRINT_CLASSES = [11, 12, 13, 14, 21, 22, 23, 30, 50, 60, 71, 83]
EPS = 1e-8


def read_window(raster, bounds, pad=1000.0):
    from rasterio.windows import from_bounds
    minx, miny, maxx, maxy = bounds
    from rasterio.windows import Window
    win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad,
                      raster.transform).round_offsets().round_lengths()
    # Clip to the raster. Without this, a window that overhangs the edge is
    # read clipped but its transform is not, and every pixel is misplaced.
    try:
        win = win.intersection(Window(0, 0, raster.width, raster.height))
    except Exception:                                      # noqa: BLE001
        raise SystemExit("The requested area does not overlap this raster. "
                         "Check that --raster covers Franconia and that "
                         "--year matches it.")
    data = raster.read(1, window=win)
    return data, raster.window_transform(win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fingerprint-only", action="store_true",
                    help="skip the polygon steps and only print per-class "
                         "NDVI at the scene dates. Use for another year "
                         "(e.g. --year 2021 with that year's raster), where "
                         "the 2022 polygons are not valid.")
    args = ap.parse_args()

    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import rowcol
    from rasterio.warp import transform as warp_xy

    rng = np.random.default_rng(args.seed)
    dates = config.DATES[args.year]
    gdf = (None if args.fingerprint_only
           else gpd.read_file(config.fields_path(args.year)))

    with rasterio.open(args.raster) as ras:
        if args.fingerprint_only:
            from rasterio.warp import transform_bounds
            bounds = transform_bounds("EPSG:4326", ras.crs, *config.BBOX)
            gdf_r = None
        else:
            gdf_r = gdf.to_crs(ras.crs)
            bounds = gdf_r.total_bounds
        cls, tf = read_window(ras, bounds)
        ras_crs = ras.crs
    print(f"Window {cls.shape}, classes present: "
          f"{sorted(int(c) for c in np.unique(cls) if c)[:30]}")

    if not args.fingerprint_only:
        # ---------------- 1. composition under the polygons ----------------
        majority, share60, share13 = [], [], []
        for geom in gdf_r.geometry:
            m = geometry_mask([geom], out_shape=cls.shape, transform=tf,
                              invert=True, all_touched=False)
            vals = cls[m]
            vals = vals[vals > 0]
            if vals.size == 0:
                majority.append(0); share60.append(np.nan); share13.append(np.nan)
                continue
            u, c = np.unique(vals, return_counts=True)
            majority.append(int(u[np.argmax(c)]))
            share60.append(float(np.mean(vals == 60)))
            share13.append(float(np.mean(vals == 13)))
        majority = np.array(majority)

        print("\n" + "=" * 66)
        print(f"1. COMPOSITION - majority v21 class under each of "
              f"{len(gdf)} polygons")
        print("=" * 66)
        u, c = np.unique(majority, return_counts=True)
        for code, n in sorted(zip(u, c), key=lambda t: -t[1])[:8]:
            print(f"  class {code:>3} {V21.get(int(code), '?'):<24} "
                  f"{n:>5} fields  {n / len(gdf):6.1%}")
        print(f"\n  mean share of pixels in class 60 (sugar beet): "
              f"{np.nanmean(share60):.1%}")
        print(f"  mean share of pixels in class 13 (winter rye): "
              f"{np.nanmean(share13):.1%}")


    # ---------------- 2. spectral fingerprint ---------------------------
    scene_crs = None
    arrays = {}
    for d in dates:
        for band in ("B04", "B08"):
            with rasterio.open(config.SCENE_DIR / d / f"{band}.tif") as s:
                arrays[(d, band)] = (s.read(1), s.transform)
                scene_crs = s.crs

    def ndvi_at(xs, ys):
        if scene_crs != ras_crs:
            xs, ys = warp_xy(ras_crs, scene_crs, list(xs), list(ys))
        out = []
        for d in dates:
            (red, t1), (nir, _) = arrays[(d, "B04")], arrays[(d, "B08")]
            r, c = rowcol(t1, xs, ys)
            r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (c >= 0) & (r < red.shape[0]) & (c < red.shape[1])
            v = np.full(len(r), np.nan)
            rv = red[r[ok], c[ok]].astype(np.float32)
            nv = nir[r[ok], c[ok]].astype(np.float32)
            off = config.boa_offset(d)
            good = (rv > 0) & (nv > 0)
            red_r, nir_r = (rv - off) / 1e4, (nv - off) / 1e4
            nd = np.where(good, (nir_r - red_r) / (nir_r + red_r + EPS),
                          np.nan)
            v[np.where(ok)[0]] = nd
            out.append(v)
        return np.stack(out, axis=1)

    print("\n" + "=" * 66)
    print("2. FINGERPRINT - mean NDVI per date for random pixels of each "
          "v21 class")
    print("=" * 66)
    labels = " ".join(f"{d[4:6]}/{d[6:]:>2}" for d in dates)
    print(f"  {'class':<28}{labels:>26}   n")

    traj = {}
    for code in FINGERPRINT_CLASSES:
        rows, cols = np.nonzero(cls == code)
        if rows.size < 50:
            continue
        pick = rng.choice(rows.size, min(args.n_sample, rows.size),
                          replace=False)
        xs, ys = rasterio.transform.xy(tf, rows[pick], cols[pick])
        nd = ndvi_at(xs, ys)
        nd = nd[np.isfinite(nd).all(axis=1)]
        if len(nd) < 30:
            continue
        traj[code] = nd.mean(axis=0)
        print(f"  {code:>3} {V21[code]:<23}"
              f"{traj[code][0]:>9.3f}{traj[code][1]:>8.3f}"
              f"{traj[code][2]:>8.3f}   {len(nd)}")

    if args.fingerprint_only:
        print("\n  (fingerprint-only: no polygon comparison)")
        return

    t = pd.read_csv(config.DATA_DIR / f"field_targets_{args.year}.csv")
    mine = t[["ndvi_apr", "ndvi_jun", "ndvi_aug"]].mean().to_numpy()
    print(f"  {'YOUR 1020 POLYGONS':<28}{mine[0]:>9.3f}{mine[1]:>8.3f}"
          f"{mine[2]:>8.3f}")

    dist = sorted((float(np.linalg.norm(v - mine)), c)
                  for c, v in traj.items())
    print("\n  closest classes to your polygons (Euclidean on 3 dates):")
    for dd, code in dist[:4]:
        print(f"    {code:>3} {V21[code]:<22} distance {dd:.3f}")

    print("\nRead: if class 60 dominates section 1 and is nearest in section "
          "2, the labels are right and the low August NDVI is crop "
          "condition or classification noise. If winter rye (13) dominates "
          "or is nearest, the polygons are NOT sugar beet and the whole "
          "benchmark's crop label must be corrected.")


if __name__ == "__main__":
    main()