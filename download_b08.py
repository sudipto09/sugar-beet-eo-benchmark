import pystac_client
import planetary_computer
import rasterio
from pathlib import Path

SCENE_DIR = "/home/sudiptochakraborty/praktikum_cv/Data/sentinel2/wuerzburg_core"
DATES_URLS = {
    "20220416": "2022-04-16",
    "20220618": "2022-06-18",
    "20220807": "2022-08-07"
}
BBOX = [9.8, 49.7, 10.3, 49.95]

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

for date_str, date_search in DATES_URLS.items():
    print(f"\nDownloading B08 for {date_str}...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{date_search}/{date_search}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    if not items:
        print(f"  No scene found for {date_str}")
        continue
    item = items[0]
    out_path = Path(f"{SCENE_DIR}/{date_str}/B08.tif")
    if out_path.exists():
        print(f"  B08 already exists")
        continue
    with rasterio.open(item.assets["B08"].href) as src:
        data = src.read(1)
        profile = src.profile
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f"  B08 ✓ shape={data.shape}")
