"""Q7: Train advanced models with hyperparameter tuning."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib


def get_default_models() -> dict:
    """Return default configurations for all models."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, random_state=42, verbose=-1,
        ),
    }


def tune_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> dict:
    """Tune XGBoost hyperparameters using Optuna."""

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBClassifier(
            **params, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params
