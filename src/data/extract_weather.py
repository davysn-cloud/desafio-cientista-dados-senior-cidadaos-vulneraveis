"""Extract weather data from Open-Meteo Historical API."""
import os
import requests
import pandas as pd


# Rio de Janeiro coordinates
RIO_LAT = -22.9068
RIO_LON = -43.1729

CACHE_PATH = "data/raw/weather_rio_2023_2024.csv"


def extract_weather(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    cache_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """Fetch daily weather data for Rio de Janeiro from Open-Meteo."""
    if os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        return pd.read_csv(cache_path)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": RIO_LAT,
        "longitude": RIO_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "windspeed_10m_max",
            "weathercode",
        ]),
        "timezone": "America/Sao_Paulo",
    }

    print("Fetching weather data from Open-Meteo...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["daily"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} days to {cache_path}")
    return df
