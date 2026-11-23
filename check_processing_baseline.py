"""
Resolve the -1000 DN offset question, once, from scene metadata.

Sentinel-2 processing baseline 04.00 (operational from 2022-01-25) shifted
L2A reflectance by +1000 DN and records BOA_ADD_OFFSET = -1000. True
reflectance is then

    rho = (DN + BOA_ADD_OFFSET) / 10000

Scenes produced under earlier baselines carry no offset. The 2021 and 2022
scenes in this project therefore sit on opposite sides of the cutover, so a
single hard-coded offset is wrong for one of the two years, and any
year-to-year comparison inherits that error.

This script asks the STAC catalogue what baseline each scene was processed
under and writes Data/scene_metadata.json, which config.boa_offset() reads.

Run once:

    python check_processing_baseline.py

Add --verify to also sample dark pixels from the downloaded rasters as an
independent check on the metadata.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import config

# Baseline 04.00 became operational on this date.
BASELINE_0400_CUTOVER = datetime(2022, 1, 25)


def parse_baseline(value):
    """'04.00' or '0400' or 4.0 -> float 4.0. None if unparseable."""
    if value is None:
        return None
    try:
        text = str(value).strip()
        if "." in text:
            return float(text)
        # '0400' style
        return float(text[:2]) + float(text[2:]) / 100.0
    except (ValueError, IndexError):
        return None


def query_stac():
    """Fetch processing baseline for every configured scene date."""
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        print("pystac_client / planetary_computer not installed.")
        print("  pip install pystac-client planetary-computer")
        sys.exit(1)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    metadata = {}
    for year, dates in config.DATES.items():
        for date_str in dates:
            day = datetime.strptime(date_str, "%Y%m%d")
            nxt = (day + timedelta(days=1)).strftime("%Y-%m-%d")

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=config.BBOX,
                datetime=f"{day.strftime('%Y-%m-%d')}/{nxt}",
            )
            items = list(search.items())
            if not items:
                print(f"  {date_str}: NO SCENE FOUND - check the date in "
                      f"config.DATES")
                continue

            item = min(items,
                       key=lambda i: i.properties.get("eo:cloud_cover", 100))
            props = item.properties

            baseline = parse_baseline(
                props.get("s2:processing_baseline")
                or props.get("processing:version"))

            if baseline is None:
                # Metadata absent: fall back to the cutover date.
                baseline = 4.0 if day >= BASELINE_0400_CUTOVER else 3.0
                source = "inferred from acquisition date"
            else:
                source = "scene metadata"

            offset = 1000.0 if baseline >= 4.0 else 0.0

            metadata[date_str] = {
                "scene_id": item.id,
                "datetime": item.datetime.isoformat(),
                "processing_baseline": baseline,
                "boa_offset": offset,
                "offset_source": source,
                "cloud_cover": props.get("eo:cloud_cover"),
            }

            print(f"  {date_str}  baseline {baseline:>5.2f}  "
                  f"offset {offset:>6.0f} DN  ({source})")

    return metadata


def verify_against_rasters(metadata):
    """Independent sanity check from the pixels themselves.

    Under baseline >= 04.00 the +1000 shift means genuinely dark surfaces
    (deep water, deep shadow) bottom out near 1000 DN rather than near 0.
    A 1st-percentile blue reflectance well above zero is consistent with
    an offset; close to zero is consistent with none.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        print("\n[skip] rasterio/numpy not available for raster verification.")
        return

    print("\nIndependent check - 1st percentile of blue band (B02):")
    print("  high (~700-1200 DN) => offset applied by the producer")
    print("  low  (~0-200 DN)    => no offset\n")

    for date_str, entry in sorted(metadata.items()):
        path = config.SCENE_DIR / date_str / "B02.tif"
        if not path.exists():
            print(f"  {date_str}: not downloaded yet ({path.name})")
            continue

        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")

        valid = data[data > 0]
        if valid.size == 0:
            print(f"  {date_str}: raster is empty")
            continue

        p1 = float(np.percentile(valid, 1))
        expected = entry["boa_offset"] > 0
        looks_offset = p1 > 500
        verdict = "consistent" if looks_offset == expected else "MISMATCH"

        print(f"  {date_str}: p1 = {p1:>7.1f} DN   "
              f"metadata says offset {entry['boa_offset']:.0f}   {verdict}")

        if verdict == "MISMATCH":
            print("      ^ investigate before extracting features. The "
                  "catalogue may already have harmonised this scene.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="also sample downloaded rasters as a cross-check")
    args = ap.parse_args()

    config.describe()
    print("Querying STAC for processing baselines...\n")

    metadata = query_stac()
    if not metadata:
        print("\nNothing resolved. Check config.DATES and network access.")
        sys.exit(1)

    if args.verify:
        verify_against_rasters(metadata)

    config.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nWritten: {config.METADATA_PATH}")

    offsets = {d: e["boa_offset"] for d, e in metadata.items()}
    if len(set(offsets.values())) > 1:
        print("\nNOTE: the scenes do NOT share one offset.")
        print("Any comparison across years must apply these per scene:")
        for date_str, off in sorted(offsets.items()):
            print(f"    {date_str}: subtract {off:.0f} DN")


if __name__ == "__main__":
    main()