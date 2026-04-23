"""Extract Brazilian public holidays from Nager.Date API."""
import os
import requests
import pandas as pd


CACHE_PATH = "data/raw/holidays_br_2023_2024.csv"


def extract_holidays(
    years: list[int] | None = None,
    cache_path: str = CACHE_PATH,
) -> pd.DataFrame:
    """Fetch Brazilian public holidays for given years."""
    if years is None:
        years = [2023, 2024]

    if os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        return pd.read_csv(cache_path)

    holidays = []
    for year in years:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/BR"
        print(f"Fetching holidays for {year}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        holidays.extend(response.json())

    df = pd.DataFrame(holidays)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} holidays to {cache_path}")
    return df
