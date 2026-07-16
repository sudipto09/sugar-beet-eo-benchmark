import requests
import rasterio
import numpy as np
from pathlib import Path

WMS_URL = "https://geoservice.dlr.de/eoc/land/wms"
BBOX = "49.55,9.7,50.05,10.6"
WIDTH, HEIGHT = 3600, 2000

# Load the raster to find pixel locations for each class code
raster_path = "/home/sudiptochakraborty/praktikum_cv/Data/croptypes_franconia_2022_wms.tif"
with rasterio.open(raster_path) as src:
    data = src.read(1)

# Find one pixel location per unique class code
unique_classes = np.unique(data)
print(f"Found {len(unique_classes)} unique class codes: {unique_classes}")

class_map = {}

for class_code in unique_classes:
    if class_code == 0:
        class_map[0] = "no data / background"
        continue
    
    # Find pixel indices for this class
    rows, cols = np.where(data == class_code)
    
    # Use the middle occurrence
    mid = len(rows) // 2
    row, col = rows[mid], cols[mid]
    
    # Query GetFeatureInfo at this pixel
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": "CROPTYPES_DE_P1Y_V02",
        "QUERY_LAYERS": "CROPTYPES_DE_P1Y_V02",
        "CRS": "EPSG:4326",
        "BBOX": BBOX,
        "WIDTH": str(WIDTH),
        "HEIGHT": str(HEIGHT),
        "FORMAT": "image/geotiff",
        "INFO_FORMAT": "application/json",
        "TIME": "2022-10-31T23:59:59.000Z",
        "I": str(col),
        "J": str(row),
        "FEATURE_COUNT": "1",
    }
    
    try:
        resp = requests.get(WMS_URL, params=params, timeout=10)
        data_json = resp.json()
        
        if data_json.get("features"):
            props = data_json["features"][0]["properties"]
            name = list(props.values())[0].strip()
        else:
            name = "unknown"
    except Exception as e:
        name = f"error: {e}"
    
    class_map[class_code] = name
    print(f"  Class {class_code:2d} → {name}")

print("\n=== COMPLETE CLASS LEGEND ===")
for code, name in sorted(class_map.items()):
    print(f"  {code:2d}: {name}")

# Save to file
import json
with open("/home/sudiptochakraborty/praktikum_cv/Data/class_legend.json", "w") as f:
    json.dump(class_map, f, indent=2)
print("\nSaved to Data/class_legend.json")
