
import requests
from pathlib import Path

Path("/home/sudiptochakraborty/praktikum_cv/Data").mkdir(parents=True, exist_ok=True)

WMS_URL = "https://geoservice.dlr.de/eoc/land/wms"

params = {
    "SERVICE": "WMS",
    "VERSION": "1.3.0",
    "REQUEST": "GetMap",
    "LAYERS": "CROPTYPES_DE_P1Y_V02",
    "CRS": "EPSG:4326",
    "BBOX": "49.55,9.7,50.05,10.6",
    "WIDTH": "3600",
    "HEIGHT": "2000",
    "FORMAT": "image/geotiff",
    "TIME": "2022-10-31T23:59:59.000Z",
}

resp = requests.get(WMS_URL, params=params, stream=True)
print("Final URL:", resp.url)
print("Status:", resp.status_code)
print("Content-Type:", resp.headers.get("Content-Type"))

out_path = "/home/sudiptochakraborty/praktikum_cv/Data/croptypes_franconia_2022_wms.tif"
if "tiff" in resp.headers.get("Content-Type", "").lower():
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    print("Saved:", out_path)
else:
    print("Non-tiff response:")
    print(resp.text[:1000])
