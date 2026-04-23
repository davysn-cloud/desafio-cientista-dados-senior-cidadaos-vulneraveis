"""Model evaluation and comparison utilities."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
)


def plot_roc_curves(models_results: dict, save_path: str = "results/figures/q7_roc_curves.png"):
    """Plot overlaid ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, result in models_results.items():
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_proba"])
        auc = roc_auc_score(result["y_true"], result["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC - Comparacao de Modelos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(models_results: dict, save_path: str = "results/figures/q7_pr_curves.png"):
    """Plot overlaid Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, result in models_results.items():
        precision, recall, _ = precision_recall_curve(result["y_true"], result["y_proba"])
        ap = average_precision_score(result["y_true"], result["y_proba"])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall - Comparacao de Modelos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_comparison_table(models_metrics: dict) -> pd.DataFrame:
    """Create a comparison DataFrame of all model metrics."""
    rows = []
    for name, metrics in models_metrics.items():
        rows.append({
            "Modelo": name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "AUC-ROC": f"{metrics['auc_roc']:.4f}",
            "AUC-PR": f"{metrics['auc_pr']:.4f}",
        })
    return pd.DataFrame(rows)
