"""
Is the radiometry sound? Scene-wide NDVI, independent of the field polygons.

"""
import argparse
import numpy as np
import rasterio
import config

EPS = 1e-8


def ndvi(red, nir, offset):
    r = (red - offset) / 10000.0
    n = (nir - offset) / 10000.0
    ok = (red > 0) & (nir > 0)
    return ((n - r) / (n + r + EPS))[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample every Nth pixel to keep it fast")
    args = ap.parse_args()

    for label, date in zip(config.DATE_LABELS, config.DATES[args.year]):
        d = config.SCENE_DIR / date
        with rasterio.open(d / "B04.tif") as a, rasterio.open(d / "B08.tif") as b:
            red = a.read(1)[::args.stride, ::args.stride].astype(np.float32)
            nir = b.read(1)[::args.stride, ::args.stride].astype(np.float32)
        off = config.boa_offset(date)
        print(f"\n{label} {date}  (verified offset {off:.0f})")
        for tag, o in (("verified", off), ("offset 0", 0.0), ("offset 2000", 2000.0)):
            v = ndvi(red, nir, o)
            p = np.percentile(v, [50, 90, 95, 99])
            print(f"  {tag:<12} NDVI median {p[0]:.3f}  p90 {p[1]:.3f}  "
                  f"p95 {p[2]:.3f}  p99 {p[3]:.3f}")
        ok = (red > 0) & (nir > 0)
        print(f"  median red {np.median((red[ok]-off)/1e4):.3f}  "
              f"median NIR {np.median((nir[ok]-off)/1e4):.3f}")
    print("\nRead: verified-offset p99 well below ~0.8 in June => "
          "radiometric problem. p99 around 0.85-0.9 => radiometry is fine.")


if __name__ == "__main__":
    main()