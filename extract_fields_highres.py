import rasterio
from rasterio.features import shapes
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

SUGAR_BEET_CLASS = 13
raster_path = "/home/sudiptochakraborty/praktikum_cv/Data/croptypes_wuerzburg_core_2022_hires.tif"

print("Loading high-res raster...")
with rasterio.open(raster_path) as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs
    print(f"Shape: {data.shape}")
    print(f"CRS: {crs}")

n_pixels = (data == SUGAR_BEET_CLASS).sum()
print(f"Sugar beet pixels: {n_pixels} ({100*n_pixels/data.size:.2f}%)")

# Vectorize
mask = (data == SUGAR_BEET_CLASS).astype(np.uint8)
polygons = []
for geom, val in shapes(mask, mask=mask, transform=transform):
    if val == 1:
        polygons.append(shape(geom))

print(f"Raw patches: {len(polygons)}")

gdf = gpd.GeoDataFrame(
    {"field_id": [f"SB_FR_{i:04d}" for i in range(len(polygons))],
     "crop_class": 13,
     "crop_name": "sugar beet",
     "region": "franconia",
     "year": 2022},
    geometry=polygons,
    crs=crs
)


gdf_utm = gdf.to_crs("EPSG:32632")
gdf["area_ha"] = gdf_utm.geometry.area / 10000  # m² → ha

print(f"\nArea distribution before filter:")
print(f"  <0.5 ha: {(gdf['area_ha'] < 0.5).sum()}")
print(f"  0.5-5 ha: {((gdf['area_ha'] >= 0.5) & (gdf['area_ha'] < 5)).sum()}")
print(f"  5-50 ha: {((gdf['area_ha'] >= 5) & (gdf['area_ha'] < 50)).sum()}")
print(f"  >50 ha: {(gdf['area_ha'] >= 50).sum()}")

# Keep real agricultural fields: 0.5 ha to 100 ha
gdf = gdf[(gdf["area_ha"] >= 0.5) & (gdf["area_ha"] <= 100)]
print(f"\nAfter filter: {len(gdf)} real sugar beet fields")
print(f"Area stats: min={gdf['area_ha'].min():.1f} ha, max={gdf['area_ha'].max():.1f} ha, mean={gdf['area_ha'].mean():.1f} ha")

# Save
out_path = "/home/sudiptochakraborty/praktikum_cv/Data/sugar_beet_fields_franconia_hires.geojson"
gdf.to_file(out_path, driver="GeoJSON")
print(f"\nSaved {len(gdf)} fields to: {out_path}")
