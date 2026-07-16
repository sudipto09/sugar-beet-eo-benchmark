import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

Path("/home/sudiptochakraborty/praktikum_cv/Data/dwd").mkdir(parents=True, exist_ok=True)

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
STATION_ID = "05705"

# ============================================================
# 1. TEMPERATURE (TU)
# ============================================================
print("Downloading temperature data for Würzburg (station 05705)...")
temp_url = f"{BASE_URL}/air_temperature/historical/stundenwerte_TU_{STATION_ID}_19480101_20251231_hist.zip"
resp = requests.get(temp_url, stream=True, timeout=120)
print(f"Temperature: Status {resp.status_code}, Size: {len(resp.content)/1024/1024:.1f} MB")

with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
    # Find the data file (starts with "produkt_")
    data_files = [f for f in z.namelist() if f.startswith("produkt_")]
    print(f"Files in zip: {z.namelist()}")
    
    with z.open(data_files[0]) as f:
        temp_df = pd.read_csv(f, sep=";", encoding="latin1")
        temp_df.columns = temp_df.columns.str.strip()
        print(f"Temperature columns: {list(temp_df.columns)}")
        print(f"Temperature rows: {len(temp_df)}")

# ============================================================
# 2. HUMIDITY (RF)
# ============================================================
print("\nFinding humidity file name...")
resp_list = requests.get(f"{BASE_URL}/relative_humidity/historical/")
lines = [l for l in resp_list.text.split('\n') if STATION_ID in l and 'hist.zip' in l]
print("Humidity files found:")
for l in lines[:5]:
    print(" ", l.strip())
