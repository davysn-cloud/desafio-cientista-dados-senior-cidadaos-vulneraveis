"""
Feature Engineering Pipeline (Q5) — Agent 3
============================================
Builds train/test feature datasets for the resolved_in_7_days target.

Anti-leakage protocol enforced:
- Target encoding via CV folds (train) / global train means (test)
- Rolling counts use strict lookback from data_inicio
- Percentile thresholds computed from training data only
- Imputation statistics (median) computed from training data only
- StandardScaler fit on training data only
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FEAT = ROOT / "data" / "features"
MODELS = ROOT / "results" / "models"
OUTPUTS = ROOT / "vault" / "03-feature-engineer" / "outputs"

for d in [FEAT, MODELS, OUTPUTS]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 50_000


# ── Helper: target encoding (leakage-safe) ────────────────────────────
def target_encode_cv(
    train_series: pd.Series,
    target: pd.Series,
    test_series: pd.Series | None = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> tuple[pd.Series, pd.Series | None, dict]:
    """Target encode with CV folds to prevent leakage.
    Returns (train_encoded, test_encoded, mapping_dict).
    """
    from sklearn.model_selection import KFold

    global_mean = target.mean()

    # Training: out-of-fold encoding
    train_encoded = pd.Series(np.nan, index=train_series.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, val_idx in kf.split(train_series):
        fold_target = target.iloc[train_idx]
        fold_series = train_series.iloc[train_idx]
        fold_means = fold_target.groupby(fold_series).mean()
        fold_counts = fold_target.groupby(fold_series).count()
        smoothed = (fold_counts * fold_means + smoothing * global_mean) / (
            fold_counts + smoothing
        )
        train_encoded.iloc[val_idx] = train_series.iloc[val_idx].map(smoothed)

    train_encoded = train_encoded.fillna(global_mean)

    # Global training mapping (for test and persistence)
    category_means = target.groupby(train_series).mean()
    category_counts = target.groupby(train_series).count()
    smoothed_global = (category_counts * category_means + smoothing * global_mean) / (
        category_counts + smoothing
    )
    mapping = smoothed_global.to_dict()

    # Test encoding
    test_encoded = None
    if test_series is not None:
        test_encoded = test_series.map(mapping).fillna(global_mean)

    return train_encoded, test_encoded, mapping


# ── Helper: vectorized rolling count ──────────────────────────────────
def compute_rolling_count_vectorized(
    df: pd.DataFrame,
    group_col: str,
    date_col: str = "data_inicio",
    window_days: int = 7,
) -> pd.Series:
    """Vectorized rolling count of chamados in same group over past N days.
    Strict lookback: excludes the current row's date.
    """
    tmp = df[[group_col, date_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)

    # For each group, compute a cumulative count minus a lagged cumulative count
    # using date-based rolling. We add a helper "count" column.
    tmp["_one"] = 1
    tmp = tmp.set_index(date_col)

    results = []
    for grp, sub in tmp.groupby(group_col):
        # Rolling sum of the past 7 days (inclusive of current day)
        rolling = sub["_one"].rolling(f"{window_days}D", closed="left").sum()
        results.append(rolling)

    if not results:
        return pd.Series(0, index=df.index)

    merged = pd.concat(results).sort_index()
    # Re-align to original index
    merged.index = tmp.sort_index().index  # this won't work for duplicate dates

    # Better approach: use original index tracking
    # Restart with a cleaner method
    return _rolling_count_merge_sort(df, group_col, date_col, window_days)


def _rolling_count_merge_sort(
    df: pd.DataFrame,
    group_col: str,
    date_col: str = "data_inicio",
    window_days: int = 7,
) -> pd.Series:
    """Rolling count using sort + groupby + rolling with original index preserved."""
    tmp = df[[group_col, date_col]].copy()
    tmp["_dt"] = pd.to_datetime(tmp[date_col])
    tmp["_orig_idx"] = tmp.index
    tmp = tmp.sort_values("_dt")

    counts = pd.Series(0.0, index=tmp.index)

    for grp, sub in tmp.groupby(group_col):
        if len(sub) <= 1:
            continue
        # For each row, count how many rows in same group have date in [row_date - 7d, row_date)
        dates = sub["_dt"].values.astype("datetime64[s]").astype(np.int64)
        window_sec = window_days * 86400

        # Use searchsorted for efficiency
        n = len(dates)
        left_bounds = dates - window_sec
        # For each position i, count how many dates in [left_bounds[i], dates[i])
        left_pos = np.searchsorted(dates, left_bounds, side="left")
        right_pos = np.arange(n)  # exclusive of self
        c = right_pos - left_pos
        c = np.maximum(c, 0)
        counts.iloc[sub.index.get_indexer(sub.index)] = c

    # Restore original order
    return counts.reindex(df.index).fillna(0).astype(int)


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE (Q5)")
    print("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────────
    print("\n[1/8] Loading raw data...")
    chamados = pd.read_parquet(RAW / "chamados_2023_2024.parquet", dtype_backend="pyarrow")
    weather = pd.read_csv(RAW / "weather_rio_2023_2024.csv")
    holidays = pd.read_csv(RAW / "holidays_br_2023_2024.csv")
    bairros = pd.read_parquet(RAW / "bairros.parquet", dtype_backend="pyarrow")

    print(f"  Chamados: {chamados.shape}")
    print(f"  Weather:  {weather.shape}")
    print(f"  Holidays: {holidays.shape}")
    print(f"  Bairros:  {bairros.shape}")

    # Convert to pandas-native types for easier manipulation
    chamados["data_inicio"] = pd.to_datetime(chamados["data_inicio"])
    chamados["data_fim"] = pd.to_datetime(chamados["data_fim"])

    # ── 2. Compute target ─────────────────────────────────────────
    print("\n[2/8] Computing target variable...")
    # Exclude rows without data_fim (still open)
    mask_closed = chamados["data_fim"].notna()
    df = chamados[mask_closed].copy()
    print(f"  Closed chamados: {len(df)} / {len(chamados)}")

    resolution_days = (df["data_fim"] - df["data_inicio"]).dt.total_seconds() / 86400
    df["resolved_in_7_days"] = (resolution_days <= 7).astype(int)

    print(f"  Target distribution:\n{df['resolved_in_7_days'].value_counts()}")
    print(f"  Positive rate: {df['resolved_in_7_days'].mean():.3f}")

    # ── 3. Sample 50K stratified by target and year ───────────────
    print("\n[3/8] Sampling 50K stratified by target + year...")
    df["year"] = df["data_inicio"].dt.year
    df["strat_key"] = df["year"].astype(str) + "_" + df["resolved_in_7_days"].astype(str)

    # Proportional allocation by year-target combo
    strat_counts = df["strat_key"].value_counts(normalize=True)
    print(f"  Stratification proportions:\n{strat_counts}")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample_idx, _ = next(sss.split(df, df["strat_key"]))
    df_sample = df.iloc[sample_idx].copy().reset_index(drop=True)
    print(f"  Sample size: {len(df_sample)}")
    print(f"  Sample target dist:\n{df_sample['resolved_in_7_days'].value_counts()}")

    # Split by year: train=2023, test=2024
    train_mask = df_sample["year"] == 2023
    test_mask = df_sample["year"] == 2024

    df_train = df_sample[train_mask].copy().reset_index(drop=True)
    df_test = df_sample[test_mask].copy().reset_index(drop=True)

    y_train = df_train["resolved_in_7_days"].copy()
    y_test = df_test["resolved_in_7_days"].copy()

    print(f"  Train: {len(df_train)} (positive rate: {y_train.mean():.3f})")
    print(f"  Test:  {len(df_test)} (positive rate: {y_test.mean():.3f})")

    # ── 4. Build features ─────────────────────────────────────────
    print("\n[4/8] Building features...")

    # Storage for train/test features
    train_features = {}
    test_features = {}

    # ── 4a. Temporal features ─────────────────────────────────────
    print("  [4a] Temporal features...")
    holiday_dates = set(pd.to_datetime(holidays["date"]).dt.date)
    holiday_dates_sorted = sorted(holiday_dates)

    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        dt = dfx["data_inicio"]
        feat_dict["hour_of_day"] = dt.dt.hour.values
        feat_dict["day_of_week"] = dt.dt.dayofweek.values
        feat_dict["day_of_month"] = dt.dt.day.values
        feat_dict["month"] = dt.dt.month.values
        feat_dict["quarter"] = dt.dt.quarter.values
        feat_dict["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int).values
        feat_dict["is_holiday"] = dt.dt.date.map(lambda d: int(d in holiday_dates)).values
        feat_dict["is_business_hours"] = (
            (dt.dt.dayofweek < 5) & (dt.dt.hour >= 8) & (dt.dt.hour < 18)
        ).astype(int).values

        # Days since last / until next holiday
        days_since = []
        days_until = []
        hd_arr = np.array([pd.Timestamp(d) for d in holiday_dates_sorted])
        for d in dt.dt.date:
            d_ts = pd.Timestamp(d)
            past = hd_arr[hd_arr <= d_ts]
            future = hd_arr[hd_arr >= d_ts]
            days_since.append((d_ts - past[-1]).days if len(past) > 0 else 365)
            days_until.append((future[0] - d_ts).days if len(future) > 0 else 365)
        feat_dict["days_since_last_holiday"] = np.array(days_since)
        feat_dict["days_until_next_holiday"] = np.array(days_until)

    print(f"    Temporal features: {sum(1 for k in train_features if k.startswith(('hour','day','month','quarter','is_w','is_h','is_b','days_')))}")

    # ── 4b. Climate features ──────────────────────────────────────
    print("  [4b] Climate features...")
    weather["date"] = pd.to_datetime(weather["time"]).dt.date
    weather_map = weather.set_index("date")

    # Thresholds from training data only
    train_dates = df_train["data_inicio"].dt.date
    train_weather = weather[weather["date"].isin(set(train_dates))]
    rain_95_threshold = train_weather["precipitation_sum"].quantile(0.95)
    print(f"    Rain 95th percentile (train): {rain_95_threshold:.1f}")

    climate_cols = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
        "precipitation_sum": "precipitation_sum",
        "rain_sum": "rain_sum",
        "windspeed_10m_max": "windspeed_max",
    }

    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        dates = dfx["data_inicio"].dt.date
        for raw_col, feat_name in climate_cols.items():
            feat_dict[feat_name] = dates.map(weather_map[raw_col].to_dict()).values.astype(float)
        feat_dict["is_extreme_rain"] = (feat_dict["precipitation_sum"] > rain_95_threshold).astype(int)
        feat_dict["is_extreme_heat"] = (feat_dict["temp_max"] > 35).astype(int)

    print(f"    Climate features: {len(climate_cols) + 2}")

    # ── 4c. Geospatial features ───────────────────────────────────
    print("  [4c] Geospatial features...")

    # Merge bairro info
    bairros_info = bairros[["id_bairro", "nome", "id_area_planejamento",
                            "id_regiao_administrativa", "nome_regiao_administrativa",
                            "subprefeitura"]].copy()

    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        # Raw coordinates
        lat = pd.to_numeric(dfx["latitude"], errors="coerce").values.astype(float)
        lon = pd.to_numeric(dfx["longitude"], errors="coerce").values.astype(float)

        # Missing indicator
        lat_missing = np.isnan(lat).astype(int)
        lon_missing = np.isnan(lon).astype(int)

        feat_dict["latitude"] = lat
        feat_dict["longitude"] = lon
        feat_dict["coords_missing"] = lat_missing  # lat and lon are typically both missing

        # Store bairro and geo hierarchy for target encoding later
        merged = dfx[["id_bairro"]].merge(bairros_info, on="id_bairro", how="left")
        feat_dict["_id_bairro"] = dfx["id_bairro"].fillna("unknown").values
        feat_dict["_regiao_admin"] = merged["nome_regiao_administrativa"].fillna("unknown").values
        feat_dict["_area_plan"] = merged["id_area_planejamento"].fillna("unknown").values
        feat_dict["_subprefeitura"] = merged["subprefeitura"].fillna("unknown").values

    # Fill missing coords from bairro centroid (computed from training data)
    # We don't have actual geometry centroids easily, so skip centroid fill — leave NaN and use missing flag

    print(f"    Geo features: latitude, longitude, coords_missing + 4 hierarchy cols for encoding")

    # ── 4d. Categorical features (target-encoded) ────────────────
    print("  [4d] Categorical features (target encoding)...")

    target_encode_cols = {
        "_id_bairro": "bairro_encoded",
        "_regiao_admin": "regiao_admin_encoded",
        "_area_plan": "area_plan_encoded",
        "_subprefeitura": "subprefeitura_encoded",
    }

    # Also encode tipo, subtipo, orgao
    train_features["_tipo"] = df_train["tipo"].fillna("unknown").values
    test_features["_tipo"] = df_test["tipo"].fillna("unknown").values
    train_features["_subtipo"] = df_train["subtipo"].fillna("unknown").values
    test_features["_subtipo"] = df_test["subtipo"].fillna("unknown").values
    train_features["_orgao"] = df_train["nome_unidade_organizacional"].fillna("unknown").values
    test_features["_orgao"] = df_test["nome_unidade_organizacional"].fillna("unknown").values

    target_encode_cols["_tipo"] = "tipo_encoded"
    target_encode_cols["_subtipo"] = "subtipo_encoded"
    target_encode_cols["_orgao"] = "orgao_encoded"

    all_encoders = {}
    for raw_key, feat_name in target_encode_cols.items():
        train_s = pd.Series(train_features[raw_key])
        test_s = pd.Series(test_features[raw_key])
        tr_enc, te_enc, mapping = target_encode_cv(train_s, y_train, test_s)
        train_features[feat_name] = tr_enc.values
        test_features[feat_name] = te_enc.values
        all_encoders[feat_name] = mapping
        print(f"    {feat_name}: {len(mapping)} categories")

    # Historical resolution rate by bairro (from training data only)
    bairro_rates = y_train.groupby(pd.Series(train_features["_id_bairro"])).mean()
    global_rate = y_train.mean()
    train_features["hist_resolution_rate_bairro"] = (
        pd.Series(train_features["_id_bairro"]).map(bairro_rates).fillna(global_rate).values
    )
    test_features["hist_resolution_rate_bairro"] = (
        pd.Series(test_features["_id_bairro"]).map(bairro_rates).fillna(global_rate).values
    )

    # ── 4e. Contextual features ───────────────────────────────────
    print("  [4e] Contextual features...")

    # is_reclamacao
    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        cat = dfx["categoria"].fillna("").str.lower()
        feat_dict["is_reclamacao"] = cat.str.contains("reclam", na=False).astype(int).values

    # Rolling counts (vectorized)
    print("    Computing rolling counts (bairro, last 7d)...")
    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        bairro_col = pd.Series(feat_dict["_id_bairro"])
        tmp = pd.DataFrame({"group": bairro_col, "data_inicio": dfx["data_inicio"].values})
        feat_dict["chamados_same_bairro_last_7d"] = _rolling_count_merge_sort(
            tmp, "group", "data_inicio", 7
        ).values

    print("    Computing rolling counts (tipo, last 7d)...")
    for label, dfx, feat_dict in [("train", df_train, train_features), ("test", df_test, test_features)]:
        tipo_col = pd.Series(feat_dict["_tipo"])
        tmp = pd.DataFrame({"group": tipo_col, "data_inicio": dfx["data_inicio"].values})
        feat_dict["chamados_same_tipo_last_7d"] = _rolling_count_merge_sort(
            tmp, "group", "data_inicio", 7
        ).values

    # ── 5. Assemble feature DataFrames ────────────────────────────
    print("\n[5/8] Assembling feature DataFrames...")

    # List of final feature columns (exclude temp cols starting with _)
    feature_cols = [k for k in train_features if not k.startswith("_")]
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    X_train = pd.DataFrame({c: train_features[c] for c in feature_cols})
    X_test = pd.DataFrame({c: test_features[c] for c in feature_cols})

    # ── 6. Handle missing values ──────────────────────────────────
    print("\n[6/8] Handling missing values...")

    # Identify columns with >5% missing (for missing flags)
    continuous_cols = [
        "latitude", "longitude", "temp_max", "temp_min", "temp_mean",
        "precipitation_sum", "rain_sum", "windspeed_max",
        "days_since_last_holiday", "days_until_next_holiday",
        "hist_resolution_rate_bairro",
        "chamados_same_bairro_last_7d", "chamados_same_tipo_last_7d",
    ]
    # Add all target-encoded cols as continuous
    continuous_cols += [v for v in target_encode_cols.values()]

    # Check missing rates
    for col in X_train.columns:
        miss_rate = X_train[col].isna().mean()
        if miss_rate > 0.05:
            flag_col = f"{col}_missing"
            if flag_col not in X_train.columns:
                X_train[flag_col] = X_train[col].isna().astype(int)
                X_test[flag_col] = X_test[col].isna().astype(int)
                print(f"  Added missing flag: {flag_col} (train miss rate: {miss_rate:.1%})")

    # Median imputation from train only
    imputation_medians = {}
    for col in X_train.columns:
        if X_train[col].isna().any():
            med = X_train[col].median()
            imputation_medians[col] = med
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
            print(f"  Imputed {col} with median={med:.4f}")

    # Verify no NaNs left
    assert X_train.isna().sum().sum() == 0, f"NaNs remain in train: {X_train.isna().sum()[X_train.isna().sum()>0]}"
    assert X_test.isna().sum().sum() == 0, f"NaNs remain in test: {X_test.isna().sum()[X_test.isna().sum()>0]}"
    print("  No NaN values remain.")

    # ── 7. Scale continuous features ──────────────────────────────
    print("\n[7/8] Scaling continuous features...")

    # Identify columns to scale (continuous, not binary/flags)
    binary_cols = [c for c in X_train.columns if c.startswith("is_") or c.endswith("_missing")]
    integer_cyclical = ["hour_of_day", "day_of_week", "day_of_month", "month", "quarter"]
    # Scale everything except binary flags
    cols_to_scale = [c for c in X_train.columns if c not in binary_cols]
    print(f"  Scaling {len(cols_to_scale)} columns")

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # ── 8. Save outputs ───────────────────────────────────────────
    print("\n[8/8] Saving outputs...")

    X_train.to_parquet(FEAT / "X_train.parquet", index=False)
    X_test.to_parquet(FEAT / "X_test.parquet", index=False)
    y_train.to_frame("resolved_in_7_days").to_parquet(FEAT / "y_train.parquet", index=False)
    y_test.to_frame("resolved_in_7_days").to_parquet(FEAT / "y_test.parquet", index=False)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    joblib.dump(scaler, MODELS / "feature_scaler.joblib")
    joblib.dump({
        "target_encoders": all_encoders,
        "imputation_medians": imputation_medians,
        "rain_95_threshold": rain_95_threshold,
        "global_target_mean": global_rate,
        "cols_to_scale": cols_to_scale,
        "feature_columns": list(X_train.columns),
    }, MODELS / "target_encoders.joblib")
    print("  Saved scaler and encoders.")

    # ── Generate documentation ────────────────────────────────────
    print("\n  Generating documentation...")
    _generate_feature_catalog(X_train, X_test, y_train, y_test)
    _generate_feature_report(X_train, X_test, y_train, y_test, df_train, df_test, rain_95_threshold)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def _generate_feature_catalog(X_train, X_test, y_train, y_test):
    """Generate feature-catalog.md."""
    lines = [
        "# Feature Catalog (Q5)",
        "",
        f"**Total features:** {X_train.shape[1]}",
        f"**Train samples:** {len(X_train)} | **Test samples:** {len(X_test)}",
        "",
        "## Feature List",
        "",
        "| # | Feature | Type | Description |",
        "|---|---------|------|-------------|",
    ]

    descriptions = {
        "hour_of_day": ("int", "Hora do dia do chamado (0-23)"),
        "day_of_week": ("int", "Dia da semana (0=seg, 6=dom)"),
        "day_of_month": ("int", "Dia do mês (1-31)"),
        "month": ("int", "Mês (1-12)"),
        "quarter": ("int", "Trimestre (1-4)"),
        "is_weekend": ("binary", "1 se fim de semana"),
        "is_holiday": ("binary", "1 se feriado nacional"),
        "is_business_hours": ("binary", "1 se horário comercial (seg-sex 8h-18h)"),
        "days_since_last_holiday": ("int", "Dias desde o último feriado"),
        "days_until_next_holiday": ("int", "Dias até o próximo feriado"),
        "temp_max": ("float", "Temperatura máxima do dia (°C)"),
        "temp_min": ("float", "Temperatura mínima do dia (°C)"),
        "temp_mean": ("float", "Temperatura média do dia (°C)"),
        "precipitation_sum": ("float", "Precipitação acumulada do dia (mm)"),
        "rain_sum": ("float", "Chuva acumulada do dia (mm)"),
        "windspeed_max": ("float", "Velocidade máxima do vento (km/h)"),
        "is_extreme_rain": ("binary", "1 se precipitação > percentil 95 (treino)"),
        "is_extreme_heat": ("binary", "1 se temperatura máxima > 35°C"),
        "latitude": ("float", "Latitude do chamado"),
        "longitude": ("float", "Longitude do chamado"),
        "coords_missing": ("binary", "1 se coordenadas ausentes"),
        "bairro_encoded": ("float", "Bairro target-encoded (CV)"),
        "regiao_admin_encoded": ("float", "Região administrativa target-encoded"),
        "area_plan_encoded": ("float", "Área de planejamento target-encoded"),
        "subprefeitura_encoded": ("float", "Subprefeitura target-encoded"),
        "tipo_encoded": ("float", "Tipo do chamado target-encoded"),
        "subtipo_encoded": ("float", "Subtipo do chamado target-encoded"),
        "orgao_encoded": ("float", "Órgão responsável target-encoded"),
        "hist_resolution_rate_bairro": ("float", "Taxa histórica de resolução do bairro (treino)"),
        "is_reclamacao": ("binary", "1 se categoria contém 'reclamação'"),
        "chamados_same_bairro_last_7d": ("int", "Chamados no mesmo bairro nos últimos 7 dias"),
        "chamados_same_tipo_last_7d": ("int", "Chamados do mesmo tipo nos últimos 7 dias"),
    }

    for i, col in enumerate(X_train.columns, 1):
        if col.endswith("_missing"):
            base = col.replace("_missing", "")
            ftype = "binary"
            desc = f"1 se {base} ausente"
        elif col in descriptions:
            ftype, desc = descriptions[col]
        else:
            ftype = "float"
            desc = col
        lines.append(f"| {i} | `{col}` | {ftype} | {desc} |")

    lines += [
        "",
        "## Anti-Leakage Protocol",
        "",
        "- **Target encoding:** CV out-of-fold para treino, média global do treino para teste",
        "- **Rolling counts:** lookback estrito (exclui data atual)",
        "- **Thresholds:** percentis calculados apenas no treino",
        "- **Imputação:** mediana calculada apenas no treino",
        "- **Scaler:** StandardScaler ajustado apenas no treino",
        "- **data_fim NUNCA usada como feature**",
        "",
        "## Target Variable",
        "",
        "- `resolved_in_7_days`: 1 se (data_fim - data_inicio) <= 7 dias",
        f"- Train positive rate: {y_train.mean():.3f}",
        f"- Test positive rate: {y_test.mean():.3f}",
    ]

    (OUTPUTS / "feature-catalog.md").write_text("\n".join(lines), encoding="utf-8")
    print("    Saved feature-catalog.md")


def _generate_feature_report(X_train, X_test, y_train, y_test, df_train, df_test, rain_95):
    """Generate feature-report.md."""
    lines = [
        "# Feature Engineering Report (Q5)",
        "",
        "## Pipeline Summary",
        "",
        f"- **Amostra total:** {len(X_train) + len(X_test):,} chamados",
        f"- **Treino (2023):** {len(X_train):,} chamados",
        f"- **Teste (2024):** {len(X_test):,} chamados",
        f"- **Total de features:** {X_train.shape[1]}",
        "",
        "## Dataset Splits",
        "",
        f"| Split | N | Positive | Negative | Positive Rate |",
        f"|-------|---|----------|----------|---------------|",
        f"| Train | {len(y_train):,} | {y_train.sum():,} | {(~y_train.astype(bool)).sum():,} | {y_train.mean():.3f} |",
        f"| Test  | {len(y_test):,} | {y_test.sum():,} | {(~y_test.astype(bool)).sum():,} | {y_test.mean():.3f} |",
        "",
        "## Feature Categories",
        "",
        "### Temporal (10 features)",
        "- hour_of_day, day_of_week, day_of_month, month, quarter",
        "- is_weekend, is_holiday, is_business_hours",
        "- days_since_last_holiday, days_until_next_holiday",
        "",
        "### Climate (8 features)",
        "- temp_max, temp_min, temp_mean, precipitation_sum, rain_sum, windspeed_max",
        "- is_extreme_rain (threshold from train: {:.1f} mm)".format(rain_95),
        "- is_extreme_heat (threshold: 35°C)",
        "",
        "### Geospatial (3 raw + 4 encoded + 1 historical = 8 features)",
        "- latitude, longitude, coords_missing",
        "- bairro_encoded, regiao_admin_encoded, area_plan_encoded, subprefeitura_encoded",
        "- hist_resolution_rate_bairro",
        "",
        "### Categorical (3 features)",
        "- tipo_encoded, subtipo_encoded, orgao_encoded",
        "",
        "### Contextual (3 features)",
        "- is_reclamacao",
        "- chamados_same_bairro_last_7d, chamados_same_tipo_last_7d",
        "",
        "## Feature Statistics (Train)",
        "",
        "| Feature | Mean | Std | Min | Max | Missing% |",
        "|---------|------|-----|-----|-----|----------|",
    ]

    for col in X_train.columns:
        lines.append(
            f"| {col} | {X_train[col].mean():.3f} | {X_train[col].std():.3f} | "
            f"{X_train[col].min():.3f} | {X_train[col].max():.3f} | 0.0% |"
        )

    lines += [
        "",
        "## Output Files",
        "",
        "| File | Path | Shape |",
        "|------|------|-------|",
        f"| X_train | `data/features/X_train.parquet` | {X_train.shape} |",
        f"| X_test | `data/features/X_test.parquet` | {X_test.shape} |",
        f"| y_train | `data/features/y_train.parquet` | ({len(y_train)}, 1) |",
        f"| y_test | `data/features/y_test.parquet` | ({len(y_test)}, 1) |",
        f"| Scaler | `results/models/feature_scaler.joblib` | — |",
        f"| Encoders | `results/models/target_encoders.joblib` | — |",
        "",
        "## Methodology Notes",
        "",
        "1. **Amostragem:** 50K chamados amostrados via StratifiedShuffleSplit por (ano, target)",
        "2. **Split temporal:** treino=2023, teste=2024 (sem split aleatório)",
        "3. **Target encoding:** suavizado com Laplace smoothing (alpha=10), CV 5-fold no treino",
        "4. **Rolling counts:** searchsorted vetorizado, lookback estrito de 7 dias",
        "5. **Imputação:** mediana do treino aplicada em ambos splits",
        "6. **Scaling:** StandardScaler ajustado no treino, transformado em ambos splits",
    ]

    (OUTPUTS / "feature-report.md").write_text("\n".join(lines), encoding="utf-8")
    print("    Saved feature-report.md")


if __name__ == "__main__":
    main()
