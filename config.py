"""
source for paths, bands, dates and normalisation.

"""

import json
import os
from pathlib import Path



PROJECT_ROOT = Path(os.environ.get(
    "PRAKTIKUM_ROOT", "/home/sudiptochakraborty/praktikum_cv"))

DATA_DIR   = PROJECT_ROOT / "Data"
SCENE_DIR  = DATA_DIR / "sentinel2" / "wuerzburg_core"
MODEL_DIR  = PROJECT_ROOT / "models" / "prithvi_300m"
CHECKPOINT = MODEL_DIR / "Prithvi_EO_V2_300M.pt"

FIELDS_PATH   = DATA_DIR / "sugar_beet_fields_franconia_hires.geojson"   # 2022


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


# Sentinel-2 L2A asset names
PRITHVI_BANDS_S2 = ["B02", "B03", "B04", "B8A", "B11", "B12"]


PRITHVI_BAND_LABELS = ["Blue", "Green", "Red", "NIR-narrow", "SWIR1", "SWIR2"]


BASELINE_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]

# Normalisation from Prithvi EO 2.0 300M config.json.
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]



DATES = {
    2022: ["20220416", "20220618", "20220807"],
    2021: ["20210426", "20210613", "20210814"],   # verified: 0.2%, 0.3%, 8.0% cloud
}

DATE_LABELS = ["April", "June", "August"]


# Bounding box used for all Sentinel-2 searches (Wuerzburg core).
BBOX = [9.8, 49.7, 10.3, 49.95]



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
    print(f"  no verified metadata for {date_str}; "
          f"assuming offset {fallback:.0f} from the baseline cutover date. "
          f"Run check_processing_baseline.py to confirm.")
    return fallback



IMG_SIZE = 64

# Clustering / evaluation
N_CLUSTERS = 5
N_SEEDS = 10
N_BOOTSTRAP = 200


def describe():
   
    print("-" * 68)
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  scenes       : {SCENE_DIR}")
    print(f"  prithvi bands: {' '.join(PRITHVI_BANDS_S2)}")
    print(f"                 {' '.join(PRITHVI_BAND_LABELS)}")
    print(f"  metadata     : "
          f"{'verified' if METADATA_PATH.exists() else 'NOT VERIFIED'}")
    print("-" * 68)