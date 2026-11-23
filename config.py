"""
Single source of truth for paths, bands, dates and normalisation.

Every other script imports from here. Override the project root without
editing code:

    export PRAKTIKUM_ROOT=/home/sudiptochakraborty/praktikum_cv

------------------------------------------------------------------------
BAND NAMING - READ THIS BEFORE CHANGING BANDS
------------------------------------------------------------------------
Prithvi EO 2.0's config.json lists its bands as
    ["B02", "B03", "B04", "B05", "B06", "B07"]
Those are *HLS* (Harmonized Landsat Sentinel-2) names, which follow the
Landsat-style harmonised numbering:

    HLS B02 = Blue        HLS B05 = NIR  (narrow)
    HLS B03 = Green       HLS B06 = SWIR 1
    HLS B04 = Red         HLS B07 = SWIR 2

In *Sentinel-2 L2A* naming the same six bands are:

    B02 Blue   B03 Green   B04 Red   B8A NIR-narrow   B11 SWIR1   B12 SWIR2

Sentinel-2's own B05/B06/B07 are red-edge bands (705 / 740 / 783 nm) and
are NOT what Prithvi expects. Feeding them into Prithvi's NIR/SWIR
channels - and normalising them with NIR/SWIR statistics - puts three of
six channels in the wrong part of the spectrum.

The published normalisation statistics confirm the mapping: the 4th value
(2734) is far above the 3rd (1433), which is the NIR jump of vegetation,
and values 5 and 6 then decline through the SWIR.
"""

import json
import os
from pathlib import Path

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ.get(
    "PRAKTIKUM_ROOT", "/home/sudiptochakraborty/praktikum_cv"))

DATA_DIR   = PROJECT_ROOT / "Data"
SCENE_DIR  = DATA_DIR / "sentinel2" / "wuerzburg_core"
MODEL_DIR  = PROJECT_ROOT / "models" / "prithvi_300m"
CHECKPOINT = MODEL_DIR / "Prithvi_EO_V2_300M.pt"

FIELDS_PATH   = DATA_DIR / "sugar_beet_fields_franconia_hires.geojson"   # 2022

# Sugar beet polygons are year specific (crop rotation). 2022 comes from the
# WMS layer CROPTYPES_DE_P1Y_V02 (class 13 there); other years are cut from
# the DLR V021 GeoTIFF (sugar beet = class 60) by
# make_polygons_from_croptypes.py.
FIELDS_PATHS = {
    2022: FIELDS_PATH,
    2021: DATA_DIR / "sugar_beet_fields_2021.geojson",
}


def fields_path(year):
    """Polygon file for one year, with a clear error if it is missing."""
    path = FIELDS_PATHS.get(year)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"No sugar beet polygons for {year} at {path}. Run "
            f"make_polygons_from_croptypes.py --year {year} --raster <DLR "
            f"CropTypes V021 GeoTIFF for {year}> --like {FIELDS_PATH}")
    return path
METADATA_PATH = DATA_DIR / "scene_metadata.json"

# --------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------

# Sentinel-2 L2A asset names, in the channel order Prithvi expects.
PRITHVI_BANDS_S2 = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# What those channels mean, for plots and tables.
PRITHVI_BAND_LABELS = ["Blue", "Green", "Red", "NIR-narrow", "SWIR1", "SWIR2"]

# Bands for the classical spectral / vegetation-index baselines.
# B08 is the broad NIR used by the standard NDVI definition.
BASELINE_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]

# Normalisation from Prithvi EO 2.0 300M config.json. The order matches
# PRITHVI_BANDS_S2 above.
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

# --------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------

DATES = {
    2022: ["20220416", "20220618", "20220807"],
    2021: ["20210426", "20210613", "20210814"],   # verified: 0.2%, 0.3%, 8.0% cloud
}

DATE_LABELS = ["April", "June", "August"]

# Each year uses its own polygons (see fields_path above); 2021 needs
# make_polygons_from_croptypes.py to have been run. See LIMITATIONS.md §1.
#
# LIMITATION: Acquisition dates differ by 5-10 days between years.
# Phenological stages are not aligned, and 2022 was a hot, dry season while
# 2021 was cool and wet. Treat cross-year differences as a season contrast,
# not a like-for-like replicate. See LIMITATIONS.md §2 and §8.

# Bounding box used for all Sentinel-2 searches (Wuerzburg core).
BBOX = [9.8, 49.7, 10.3, 49.95]

# --------------------------------------------------------------------
# Radiometry
# --------------------------------------------------------------------

# Sentinel-2 processing baseline 04.00 (from 2022-01-25) added a
# +1000 DN shift, reversed with BOA_ADD_OFFSET = -1000. Scenes before
# that date carry no offset. 2021 and 2022 therefore need DIFFERENT
# values, which is why this is read from measured metadata rather than
# hard-coded.
#
# Run check_processing_baseline.py to write scene_metadata.json.

_DEFAULT_OFFSETS = {2021: 0.0, 2022: 1000.0}


def boa_offset(date_str):
    """DN to subtract before scaling, for one YYYYMMDD scene.

    Prefers measured metadata; falls back to the baseline cutover date
    and says so, so an unverified number never passes silently.
    """
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        if date_str in meta:
            return float(meta[date_str]["boa_offset"])

    year = int(date_str[:4])
    fallback = _DEFAULT_OFFSETS.get(year, 0.0)
    print(f"  [warn] no verified metadata for {date_str}; "
          f"assuming offset {fallback:.0f} from the baseline cutover date. "
          f"Run check_processing_baseline.py to confirm.")
    return fallback


# --------------------------------------------------------------------
# Model input
# --------------------------------------------------------------------

# Prithvi was pretrained on 30 m HLS at 224x224. Sugar beet fields are
# small, so patches are resampled to a fixed grid. 64 keeps 4x4 patches
# per frame at patch size 16.
#
# LIMITATION: Prithvi was pretrained at 30 m; inputs are 10-20 m Sentinel-2
# resampled to 64×64. The model sees finer detail than it trained on.
# This is a systematic shift, not a per-field bias. See LIMITATIONS.md §3.
IMG_SIZE = 64

# Clustering / evaluation
N_CLUSTERS = 5
N_SEEDS = 10
N_BOOTSTRAP = 200


def describe():
    """Print the resolved configuration. Call at the top of each script."""
    print("-" * 68)
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  scenes       : {SCENE_DIR}")
    print(f"  prithvi bands: {' '.join(PRITHVI_BANDS_S2)}")
    print(f"                 {' '.join(PRITHVI_BAND_LABELS)}")
    print(f"  metadata     : "
          f"{'verified' if METADATA_PATH.exists() else 'NOT VERIFIED'}")
    print("-" * 68)