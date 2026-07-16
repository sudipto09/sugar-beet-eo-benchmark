import pystac_client
import planetary_computer
import rasterio
from pathlib import Path

BBOX  = [9.8, 49.7, 10.3, 49.95]
BANDS = ["B02","B03","B04","B05","B06","B07","B08"]

TARGET_DATES = [
    ("2021-04-01","2021-04-30"),
    ("2021-06-15","2021-07-10"),
    ("2021-08-01","2021-08-31"),
]

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

selected = []
for start, end in TARGET_DATES:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    if items:
        best = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        selected.append(best)
        print(f"{start[:7]}: {best.datetime.strftime('%Y-%m-%d')} "
              f"(cloud: {best.properties.get('eo:cloud_cover','?'):.1f}%)")
    else:
        print(f"{start[:7]}: NO SCENE FOUND")

for item in selected:
    date_str = item.datetime.strftime("%Y%m%d")
    scene_dir = Path(f"/home/sudiptochakraborty/praktikum_cv/Data/sentinel2/wuerzburg_core_2021/{date_str}")
    scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {date_str}...")
    for band in BANDS:
        out_path = scene_dir / f"{band}.tif"
        if out_path.exists():
            print(f"  {band} exists")
            continue
        with rasterio.open(item.assets[band].href) as src:
            data = src.read(1)
            profile = src.profile
        with rasterio.open(out_path,'w',**profile) as dst:
            dst.write(data, 1)
        print(f"  {band} ✓")
print("\nDone.")
