import pandas as pd
import numpy as np

# Create synthetic weather data for testing
dates = pd.date_range("2020-04-01", "2022-11-30", freq="D")
doy = dates.dayofyear

temp = 10 + 15*np.sin(2*np.pi*doy/365) + np.random.normal(0, 2, len(dates))
humidity = 60 + 20*np.sin(2*np.pi*(doy-90)/365) + np.random.normal(0, 5, len(dates))

weather_df = pd.DataFrame({
    "date": dates,
    "temperature": temp,
    "humidity": humidity
})

# Wolf & Verrert thresholds
weather_df["risk"] = (
    (weather_df["temperature"] >= 15) &
    (weather_df["temperature"] <= 28) &
    (weather_df["humidity"] >= 60)
).astype(int)

weather_df["week"] = weather_df["date"].dt.isocalendar().week
weather_df["year"] = weather_df["date"].dt.year
weekly_risk = weather_df.groupby(["year", "week"])["risk"].max()

print("Risk events by year:")
print(weather_df.groupby(weather_df["date"].dt.year)["risk"].sum())

weather_df.to_csv("/home/sudiptochakraborty/praktikum_cv/Data/weather_timeseries.csv", index=False)
weekly_risk.to_csv("/home/sudiptochakraborty/praktikum_cv/Data/risk_windows.csv")
print("Saved to /home/sudiptochakraborty/praktikum_cv/Data/")
