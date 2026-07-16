import geopandas as gpd
from shapely.geometry import box

# Create 100 test sugar beet field polygons
# (Real DLR fields come later; for now, valid geometries)

fields_data = {
    "field_id": [f"FR_{i:03d}" for i in range(50)] + [f"NRW_{i:03d}" for i in range(50)],
    "region": (["franconia"]*50) + (["nrw"]*50),
    "geometry": (
        [box(9.7 + (i%10)*0.08, 49.55 + (i//10)*0.08, 
             9.7 + (i%10)*0.08 + 0.05, 49.55 + (i//10)*0.08 + 0.05) 
         for i in range(50)] +
        [box(11.5 + (i%10)*0.08, 51.9 + (i//10)*0.08,
             11.5 + (i%10)*0.08 + 0.05, 51.9 + (i//10)*0.08 + 0.05)
         for i in range(50)]
    )
}

gdf = gpd.GeoDataFrame(fields_data, crs="EPSG:4326")

import os
os.makedirs("/home/sudiptochakraborty/practikum_cv/Data", exist_ok=True)
gdf.to_file("/home/sudiptochakraborty/practikum_cv/Data/fields.geojson", driver="GeoJSON")
print(f"Created {len(gdf)} test fields in /home/sudiptochakraborty/practikum_cv/Data/fields.geojson")
