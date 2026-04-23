# Task: Extract Weather Data (Open-Meteo API)

## Objective
Fetch daily weather data for Rio de Janeiro (2023-2024) from Open-Meteo Historical API.

## Steps
1. Call `https://archive-api.open-meteo.com/v1/archive` with Rio coordinates
2. Request daily variables: temperature_2m_max, temperature_2m_min, temperature_2m_mean,
   precipitation_sum, rain_sum, windspeed_10m_max, weathercode
3. Parse JSON response into DataFrame
4. Save to `data/raw/weather_rio_2023_2024.csv`
5. Verify: exactly 731 rows (365 + 366), no missing dates

## API Parameters
- latitude: -22.9068
- longitude: -43.1729
- start_date: 2023-01-01
- end_date: 2024-12-31
- timezone: America/Sao_Paulo
