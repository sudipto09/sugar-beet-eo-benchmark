import pystac_client
import planetary_computer
import rasterio
import numpy as np
from pathlib import Path


BBOX = [9.8, 49.7, 10.3, 49.95]

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]

# Same 4 dates we want for Prithvi
TARGET_DATES = [
    ("2022-04-01", "2022-04-30"),   # April
    ("2022-06-15", "2022-07-10"),   # June
    ("2022-08-01", "2022-08-31"),   # August
    ("2022-11-01", "2022-11-30"),   # November
]

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("Searching for Sentinel-2 scenes matching field locations...")
print(f"BBOX: {BBOX}")

selected_items = []
for start, end in TARGET_DATES:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    if items:
        # Pick lowest cloud cover
        best = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        selected_items.append(best)
        print(f"  {start[:7]}: {best.datetime.strftime('%Y-%m-%d')} "
              f"(cloud: {best.properties.get('eo:cloud_cover', '?'):.1f}%)")
    else:
        print(f"  {start[:7]}: NO SCENE FOUND")

print(f"\nDownloading {len(selected_items)} scenes...")

for item in selected_items:
    date_str = item.datetime.strftime("%Y%m%d")
    scene_dir = Path(f"/home/sudiptochakraborty/praktikum_cv/Data/sentinel2/wuerzburg_core/{date_str}")
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nScene: {date_str}")
    for band in BANDS:
        out_path = scene_dir / f"{band}.tif"
        if out_path.exists():
            print(f"  {band} already exists")
            continue
        if band in item.assets:
            try:
                with rasterio.open(item.assets[band].href) as src:
                    data = src.read(1)
                    profile = src.profile
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                print(f"  {band} ✓ (shape: {data.shape})")
            except Exception as e:
                print(f"  {band} ERROR: {e}")

print("\nDownload complete.")

# Verify coverage
print("\nVerifying scene coverage vs field locations...")
test_scene = f"/home/sudiptochakraborty/praktikum_cv/Data/sentinel2/wuerzburg_core/{selected_items[0].datetime.strftime('%Y%m%d')}/B02.tif"
with rasterio.open(test_scene) as src:
    print(f"Scene CRS: {src.crs}")
    print(f"Scene bounds: {src.bounds}")
    print(f"Scene shape: {src.shape}")

# Field location for comparison
import geopandas as gpd
gdf = gpd.read_file("/home/sudiptochakraborty/praktikum_cv/Data/sugar_beet_fields_franconia_hires.geojson")
gdf_utm = gdf.to_crs("EPSG:32632")
print(f"\nField bounds (UTM): {gdf_utm.total_bounds}")
print(f"Fields should now be INSIDE scene bounds.")
