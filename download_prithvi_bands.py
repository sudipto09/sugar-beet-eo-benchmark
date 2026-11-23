"""
Download the Sentinel-2 bands Prithvi actually expects.

"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import config

ALL_BANDS = sorted(set(config.PRITHVI_BANDS_S2) | set(config.BASELINE_BANDS))

# Month windows used when discovering scenes for a new year.
DISCOVERY_WINDOWS = [("04-01", "04-30"), ("06-10", "07-10"), ("08-01", "08-31")]


def open_catalog():
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        print("pystac_client / planetary_computer not installed.")
        print("  pip install pystac-client planetary-computer")
        sys.exit(1)

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def discover(catalog, year, max_cloud=30):
    """Find the least cloudy scene in each seasonal window of a year."""
    print(f"\nDiscovering scenes for {year} (cloud < {max_cloud}%)\n")
    found = []
    for start, end in DISCOVERY_WINDOWS:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=config.BBOX,
            datetime=f"{year}-{start}/{year}-{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = list(search.items())
        if not items:
            print(f"  {year}-{start} to {year}-{end}: NO SCENE under "
                  f"{max_cloud}% cloud")
            continue
        best = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        date_str = best.datetime.strftime("%Y%m%d")
        found.append(date_str)
        print(f"  {date_str}  cloud {best.properties.get('eo:cloud_cover', 0):.1f}%"
              f"  {best.id}")

    if found:
        print(f"\nPut these in config.DATES[{year}]:")
        print(f"    {year}: {found},")
    return found


def fetch_scene(catalog, date_str, overwrite=False):
    """Download every required band for one acquisition date."""
    day = datetime.strptime(date_str, "%Y%m%d")
    nxt = (day + timedelta(days=1)).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=config.BBOX,
        datetime=f"{day.strftime('%Y-%m-%d')}/{nxt}",
    )
    items = list(search.items())
    if not items:
        print(f"  {date_str}: NO SCENE FOUND")
        return False

    item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    scene_dir = Path(config.SCENE_DIR) / date_str
    scene_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {date_str}  {item.id}")
    print(f"    cloud {item.properties.get('eo:cloud_cover', 0):.1f}%  "
          f"baseline {item.properties.get('s2:processing_baseline', '?')}")

    import rasterio

    ok = True
    for band in ALL_BANDS:
        out_path = scene_dir / f"{band}.tif"
        if out_path.exists() and not overwrite:
            print(f"    {band:<4} already present")
            continue
        if band not in item.assets:
            print(f"    {band:<4} NOT in scene assets - skipped")
            ok = False
            continue
        try:
            with rasterio.open(item.assets[band].href) as src:
                data = src.read(1)
                profile = src.profile
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            print(f"    {band:<4} ok  {data.shape}  {src.res[0]:.0f} m")
        except Exception as exc:                      # noqa: BLE001
            print(f"    {band:<4} ERROR: {exc}")
            ok = False

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="only this year")
    ap.add_argument("--discover", type=int, metavar="YEAR",
                    help="search for suitable dates instead of downloading")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config.describe()
    catalog = open_catalog()

    if args.discover:
        discover(catalog, args.discover)
        return

    years = [args.year] if args.year else sorted(config.DATES)
    print(f"Bands: {' '.join(ALL_BANDS)}")

    all_ok = True
    for year in years:
        print(f"\n=== {year} ===")
        for date_str in config.DATES[year]:
            all_ok &= fetch_scene(catalog, date_str, args.overwrite)

    print("\nDone.")
    if not all_ok:
        print("Some bands are missing - re-run, or use --discover to find "
              "dates that carry the full set.")
    print("Next: python check_processing_baseline.py --verify")


if __name__ == "__main__":
    main()