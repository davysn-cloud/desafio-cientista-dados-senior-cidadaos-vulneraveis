"""Q9: Priority score computation."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_priority_score(
    y_proba: pd.Series,
    urgency_score: pd.Series,
    equity_score: pd.Series,
    context_score: pd.Series,
    weights: dict = None,
) -> pd.Series:
    """Compute composite priority score.

    priority_score = w1 * P(delay) + w2 * urgency + w3 * equity + w4 * context

    Args:
        y_proba: Model probability of resolving in 7 days (higher = more likely resolved)
        urgency_score: Urgency based on chamado type (normalized 0-1)
        equity_score: Territorial equity score (normalized 0-1)
        context_score: Weather/contextual score (normalized 0-1)
        weights: dict with keys w1, w2, w3, w4 (must sum to 1)

    Returns:
        Priority score normalized to [0, 1]. Higher = higher priority.
    """
    if weights is None:
        weights = {"w1": 0.40, "w2": 0.20, "w3": 0.25, "w4": 0.15}

    # P(delay) = 1 - P(resolved in 7 days)
    p_delay = 1 - y_proba

    score = (
        weights["w1"] * p_delay
        + weights["w2"] * urgency_score
        + weights["w3"] * equity_score
        + weights["w4"] * context_score
    )

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    score_normalized = scaler.fit_transform(score.values.reshape(-1, 1)).flatten()

    return pd.Series(score_normalized, index=y_proba.index, name="priority_score")


def compute_equity_score(
    bairro_series: pd.Series,
    historical_resolution_rates: dict,
) -> pd.Series:
    """Compute equity score: inverse of historical resolution rate.

    Bairros with lower resolution rates get higher equity scores,
    ensuring underserved areas receive more attention.
    """
    global_mean = np.mean(list(historical_resolution_rates.values()))
    rates = bairro_series.map(historical_resolution_rates).fillna(global_mean)
    equity = 1 - rates  # Invert: low resolution = high equity need
    return (equity - equity.min()) / (equity.max() - equity.min())
