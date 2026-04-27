import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import joblib


def train_logistic_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    # saga converge mais rápido com features esparsas/muitas categoricas
    model = LogisticRegression(max_iter=1000, random_state=42, solver="saga")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
