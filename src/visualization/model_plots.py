"""Model visualization utilities."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    save_path: str = "results/figures/q8_shap_summary_beeswarm.png",
):
    """Create SHAP beeswarm summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP - Importancia das Features")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_bar(
    model,
    X_test: pd.DataFrame,
    top_n: int = 10,
    save_path: str = "results/figures/q8_shap_bar_top10.png",
):
    """Create SHAP bar plot for top N features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=top_n, show=False)
    plt.title(f"Top {top_n} Features por Importancia SHAP")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_comparison(
    native_importance: pd.Series,
    shap_importance: pd.Series,
    top_n: int = 10,
    save_path: str = "results/figures/q8_feature_importance_comparison.png",
):
    """Compare native feature importance with SHAP importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    native_importance.nlargest(top_n).plot.barh(ax=axes[0], color="#2E86C1")
    axes[0].set_title("Importancia Nativa (Tree)")
    axes[0].set_xlabel("Importancia")

    shap_importance.nlargest(top_n).plot.barh(ax=axes[1], color="#F39C12")
    axes[1].set_title("Importancia SHAP (|mean|)")
    axes[1].set_xlabel("Mean |SHAP|")

    plt.suptitle("Comparacao de Importancia de Features", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
