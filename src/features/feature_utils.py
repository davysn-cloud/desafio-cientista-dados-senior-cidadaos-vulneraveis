"""Utility functions for feature engineering."""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def target_encode_cv(
    train_series: pd.Series,
    target: pd.Series,
    test_series: pd.Series = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> tuple[pd.Series, pd.Series | None]:
    """Target encode with CV folds to prevent leakage.

    For training: use out-of-fold means.
    For test: use global training mean per category.
    """
    global_mean = target.mean()

    # Training: out-of-fold encoding
    train_encoded = pd.Series(np.nan, index=train_series.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_series):
        fold_means = target.iloc[train_idx].groupby(train_series.iloc[train_idx]).mean()
        fold_counts = target.iloc[train_idx].groupby(train_series.iloc[train_idx]).count()
        smoothed = (fold_counts * fold_means + smoothing * global_mean) / (fold_counts + smoothing)
        train_encoded.iloc[val_idx] = train_series.iloc[val_idx].map(smoothed)

    train_encoded = train_encoded.fillna(global_mean)

    # Test: global training means
    test_encoded = None
    if test_series is not None:
        category_means = target.groupby(train_series).mean()
        category_counts = target.groupby(train_series).count()
        smoothed_global = (category_counts * category_means + smoothing * global_mean) / (category_counts + smoothing)
        test_encoded = test_series.map(smoothed_global).fillna(global_mean)

    return train_encoded, test_encoded


def compute_rolling_count(
    df: pd.DataFrame,
    group_col: str,
    date_col: str = "data_inicio",
    window_days: int = 7,
) -> pd.Series:
    """Compute rolling count of chamados in same group over past N days.

    Uses strict lookback (excludes current row) to prevent leakage.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    counts = []
    for _, row in df.iterrows():
        cutoff = row[date_col] - pd.Timedelta(days=window_days)
        mask = (
            (df[group_col] == row[group_col])
            & (df[date_col] >= cutoff)
            & (df[date_col] < row[date_col])
        )
        counts.append(mask.sum())

    return pd.Series(counts, index=df.index)
