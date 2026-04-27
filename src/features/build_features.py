import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def compute_target(df: pd.DataFrame) -> pd.Series:
    # 1 = resolvido em até 7 dias, 0 = atrasado
    df["data_inicio"] = pd.to_datetime(df["data_inicio"])
    df["data_fim"] = pd.to_datetime(df["data_fim"])
    resolution_days = (df["data_fim"] - df["data_inicio"]).dt.total_seconds() / 86400
    return (resolution_days <= 7).astype(int)


def build_temporal_features(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["data_inicio"])
    holiday_dates = set(pd.to_datetime(holidays_df["date"]).dt.date)

    features = pd.DataFrame(index=df.index)
    features["hour_of_day"] = dt.dt.hour
    features["day_of_week"] = dt.dt.dayofweek
    features["day_of_month"] = dt.dt.day
    features["month"] = dt.dt.month
    features["quarter"] = dt.dt.quarter
    features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    features["is_holiday"] = dt.dt.date.map(lambda d: int(d in holiday_dates))
    features["is_business_hours"] = (
        (dt.dt.dayofweek < 5) & (dt.dt.hour >= 8) & (dt.dt.hour < 18)
    ).astype(int)

    return features


def build_climate_features(df: pd.DataFrame, weather_df: pd.DataFrame, train_weather: pd.DataFrame = None) -> pd.DataFrame:
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["time"]).dt.date

    df_date = pd.to_datetime(df["data_inicio"]).dt.date
    date_to_idx = {d: i for i, d in enumerate(weather["date"])}

    features = pd.DataFrame(index=df.index)
    merged = df_date.map(date_to_idx)

    for col in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                 "precipitation_sum", "rain_sum", "windspeed_10m_max"]:
        if col in weather.columns:
            features[col.replace("temperature_2m_", "temp_")] = merged.map(
                dict(zip(range(len(weather)), weather[col]))
            )

    # Extreme thresholds (from training data only to prevent leakage)
    if train_weather is not None:
        rain_95 = train_weather["precipitation_sum"].quantile(0.95)
    else:
        rain_95 = weather["precipitation_sum"].quantile(0.95)

    if "precipitation_sum" in features.columns:
        features["is_extreme_rain"] = (features["precipitation_sum"] > rain_95).astype(int)
    if "temp_max" in features.columns:
        features["is_extreme_heat"] = (features["temp_max"] > 35).astype(int)

    return features


def build_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    if "latitude" in df.columns:
        features["latitude"] = df["latitude"]
    if "longitude" in df.columns:
        features["longitude"] = df["longitude"]
    return features
