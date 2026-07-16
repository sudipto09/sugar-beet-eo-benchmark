import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path

Path("/home/sudiptochakraborty/praktikum_cv/Data/dwd").mkdir(parents=True, exist_ok=True)

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
STATION_ID = "05705"

print("Downloading Würzburg weather data...")
temp_url = f"{BASE_URL}/air_temperature/historical/stundenwerte_TU_{STATION_ID}_19480101_20251231_hist.zip"
resp = requests.get(temp_url, stream=True, timeout=120)

with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
    with z.open("produkt_tu_stunde_19480101_20251231_05705.txt") as f:
        df = pd.read_csv(f, sep=";", encoding="latin1")

df.columns = df.columns.str.strip()
df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H")
df["TT_TU"] = pd.to_numeric(df["TT_TU"], errors="coerce")
df["RF_TU"] = pd.to_numeric(df["RF_TU"], errors="coerce")

# CRITICAL: Replace DWD sentinel value -999 with NaN
df.loc[df["TT_TU"] <= -99, "TT_TU"] = np.nan
df.loc[df["RF_TU"] <= -99, "RF_TU"] = np.nan

# Filter to study period
df = df[
    (df["MESS_DATUM"] >= "2020-04-01") &
    (df["MESS_DATUM"] <= "2022-11-30")
].copy()

print(f"Hourly records 2020-2022: {len(df)}")
print(f"Temperature: {df['TT_TU'].min():.1f}°C to {df['TT_TU'].max():.1f}°C")
print(f"Humidity: {df['RF_TU'].min():.1f}% to {df['RF_TU'].max():.1f}%")
print(f"Missing temp: {df['TT_TU'].isna().sum()}, missing RH: {df['RF_TU'].isna().sum()}")

df["date"] = df["MESS_DATUM"].dt.date

# Per day: mean temp + count valid hours with RH >= 60%
daily = df.groupby("date").agg(
    temp_mean=("TT_TU", "mean"),
    temp_max=("TT_TU", "max"),
    rh_mean=("RF_TU", "mean"),
    # Only count hours where RH is valid (not NaN) AND >= 60%
    rh_hours_above_60=("RF_TU", lambda x: (x >= 60).sum()),
    n_valid_hours=("TT_TU", lambda x: x.notna().sum())
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])
daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month

# Wolf & Verreet rule — only apply if we have enough valid hours
daily["infection_risk"] = (
    (daily["temp_mean"] >= 15) &        # Mean temp >= 15°C
    (daily["temp_mean"] <= 28) &        # Mean temp <= 28°C (upper limit for spore germination)
    (daily["rh_hours_above_60"] >= 6) & # RH >= 60% for at least 6 hours
    (daily["n_valid_hours"] >= 18)      # Only flag if we have >= 18 valid hourly readings
).astype(int)

# Growing season only
daily_season = daily[daily["month"].between(5, 9)].copy()

print(f"\nInfection risk by year (growing season May-Sep):")
for year in [2020, 2021, 2022]:
    yr = daily_season[daily_season["year"] == year]
    risk_days = yr["infection_risk"].sum()
    total_days = len(yr)
    pct = 100 * risk_days / total_days if total_days > 0 else 0
    print(f"  {year}: {risk_days} / {total_days} days at risk ({pct:.1f}%)")

# Weekly aggregation
daily_season = daily_season.copy()
daily_season["week"] = daily_season["date"].dt.isocalendar().week.astype(int)

weekly_risk = daily_season.groupby(["year", "week"]).agg(
    risk=("infection_risk", "max"),
    risk_days=("infection_risk", "sum"),
    temp_mean=("temp_mean", "mean"),
    rh_mean=("rh_mean", "mean")
).reset_index()

print(f"\nWeekly risk summary:")
for year in [2020, 2021, 2022]:
    yr = weekly_risk[weekly_risk["year"] == year]
    risk_weeks = yr["risk"].sum()
    print(f"  {year}: {risk_weeks} / {len(yr)} weeks at risk")

print(f"\nRisk weeks per year (should differ):")
for year in [2020, 2021, 2022]:
    yr = weekly_risk[weekly_risk["year"] == year]
    risk_wks = yr[yr["risk"] == 1]["week"].tolist()
    print(f"  {year}: weeks {risk_wks}")

# Class balance check
all_risk = weekly_risk["risk"]
print(f"\nClass balance: {all_risk.sum()} positive / {(all_risk==0).sum()} negative weeks")
print(f"Positive rate: {100*all_risk.mean():.1f}%")

# Save
daily.to_csv("/home/sudiptochakraborty/praktikum_cv/Data/dwd/daily_weather_2020_2022.csv", index=False)
weekly_risk.to_csv("/home/sudiptochakraborty/praktikum_cv/Data/risk_windows_real.csv", index=False)
print("\nSaved risk_windows_real.csv")
