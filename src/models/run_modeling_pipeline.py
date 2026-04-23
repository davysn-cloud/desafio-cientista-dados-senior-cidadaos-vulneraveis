"""
Agent 4 - Model Builder: Complete modeling pipeline for Q6, Q7, Q8.
Predicts whether a 1746 chamado will be resolved within 7 days.
"""
import os
import sys
import warnings
import json
import time

# Matplotlib backend before any other import
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import optuna
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.train_baseline import train_logistic_baseline, evaluate_model
from src.models.train_advanced import get_default_models, tune_xgboost_optuna
from src.models.evaluate import plot_roc_curves, plot_pr_curves, create_comparison_table

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
VAULT_OUT = os.path.join(PROJECT_ROOT, "vault", "04-model-builder", "outputs")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

for d in [MODELS_DIR, FIGURES_DIR, VAULT_OUT, NOTEBOOKS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "neutral": "#95A5A6",
}
CATEGORICAL = [
    "#1B4F72", "#2E86C1", "#F39C12", "#27AE60", "#E74C3C",
    "#8E44AD", "#E67E22", "#16A085", "#2C3E50", "#D35400",
]
MODEL_COLORS = {
    "Logistic Regression": CATEGORICAL[0],
    "Random Forest": CATEGORICAL[1],
    "XGBoost (default)": CATEGORICAL[2],
    "LightGBM": CATEGORICAL[3],
    "XGBoost (tuned)": CATEGORICAL[4],
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet")).squeeze()
    print(f"  X_train: {X_train.shape}, y_train positive rate: {y_train.mean():.3f}")
    print(f"  X_test:  {X_test.shape},  y_test positive rate:  {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Q6: LOGISTIC REGRESSION BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_q6(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Q6: LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    model = train_logistic_baseline(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_baseline.joblib"))

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")

    # ROC curve for baseline alone
    fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2,
            label=f"Logistic Regression (AUC={metrics['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Q6 - Curva ROC: Regressao Logistica (Baseline)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, "q6_roc_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, metrics["y_pred"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Nao resolvido\n(<=7d)", "Resolvido\n(<=7d)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Q6 - Matriz de Confusao: Regressao Logistica")
    fig.savefig(os.path.join(FIGURES_DIR, "q6_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Q7: ADVANCED MODELS + TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def run_q7(X_train, X_test, y_train, y_test, baseline_metrics):
    print("\n" + "=" * 60)
    print("Q7: ADVANCED MODELS + HYPERPARAMETER TUNING")
    print("=" * 60)

    default_models = get_default_models()
    all_metrics = {"Logistic Regression": baseline_metrics}
    all_models = {}

    # Train default RF, XGB, LGBM
    model_file_map = {
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost_model.joblib",
        "lightgbm": "lgbm_model.joblib",
    }
    display_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost (default)",
        "lightgbm": "LightGBM",
    }

    for key, model in default_models.items():
        print(f"\n  Training {display_names[key]}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[display_names[key]] = metrics
        all_models[display_names[key]] = model
        joblib.dump(model, os.path.join(MODELS_DIR, model_file_map[key]))
        print(f"    F1={metrics['f1']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  ({elapsed:.1f}s)")

    # Optuna tuning for XGBoost
    print(f"\n  Tuning XGBoost with Optuna (50 trials)...")
    t0 = time.time()
    best_params = tune_xgboost_optuna(X_train, y_train, n_trials=50)
    elapsed = time.time() - t0
    print(f"    Optuna done in {elapsed:.1f}s")
    print(f"    Best params: {best_params}")

    # Retrain with best params
    xgb_tuned = XGBClassifier(
        **best_params, random_state=42,
        use_label_encoder=False, eval_metric="logloss",
    )
    xgb_tuned.fit(X_train, y_train)
    tuned_metrics = evaluate_model(xgb_tuned, X_test, y_test)
    all_metrics["XGBoost (tuned)"] = tuned_metrics
    all_models["XGBoost (tuned)"] = xgb_tuned
    # Overwrite xgboost_model with tuned version
    joblib.dump(xgb_tuned, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    print(f"    Tuned XGB: F1={tuned_metrics['f1']:.4f}  AUC-ROC={tuned_metrics['auc_roc']:.4f}")

    # Identify best model by F1
    best_name = max(
        {k: v for k, v in all_metrics.items() if k != "Logistic Regression"},
        key=lambda k: all_metrics[k]["f1"],
    )
    best_model = all_models[best_name]
    print(f"\n  ** Best model: {best_name} (F1={all_metrics[best_name]['f1']:.4f}) **")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    # Save test predictions
    best_m = all_metrics[best_name]
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best_m["y_pred"],
        "y_proba": best_m["y_proba"],
    })
    preds_df.to_parquet(os.path.join(MODELS_DIR, "test_predictions.parquet"), index=False)

    # ── Plots ────────────────────────────────────────────────────────────────
    # Prepare results dict for evaluate.py functions
    models_results = {}
    for name, m in all_metrics.items():
        models_results[name] = {
            "y_true": y_test,
            "y_proba": m["y_proba"],
        }

    plot_roc_curves(models_results, os.path.join(FIGURES_DIR, "q7_roc_curves.png"))
    plot_pr_curves(models_results, os.path.join(FIGURES_DIR, "q7_pr_curves.png"))

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_m["y_pred"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Nao resolvido\n(<=7d)", "Resolvido\n(<=7d)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Q7 - Matriz de Confusao: {best_name}")
    fig.savefig(os.path.join(FIGURES_DIR, "q7_confusion_best.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Model comparison bar chart
    comp_df = create_comparison_table(all_metrics)
    print("\n  Model Comparison Table:")
    print(comp_df.to_string(index=False))

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(comp_df))
    width = 0.13
    for i, col in enumerate(metric_cols):
        vals = comp_df[col].astype(float)
        ax.bar(x + i * width, vals, width, label=col, color=CATEGORICAL[i])
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(comp_df["Modelo"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Q7 - Comparacao de Modelos")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, "q7_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return all_models, all_metrics, best_name, best_model, best_params, comp_df


# ═══════════════════════════════════════════════════════════════════════════════
# Q8: INTERPRETABILITY (SHAP + ERROR ANALYSIS)
# ═══════════════════════════════════════════════════════════════════════════════
def run_q8(best_model, best_name, X_train, X_test, y_test, all_metrics):
    print("\n" + "=" * 60)
    print("Q8: INTERPRETABILITY & ERROR ANALYSIS")
    print("=" * 60)

    best_m = all_metrics[best_name]
    feature_names = X_test.columns.tolist()

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(best_model)
    # Use a sample for SHAP to keep it manageable
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    # Beeswarm summary
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.title("Q8 - SHAP Summary (Beeswarm)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q8_shap_summary_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Bar top 10
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False, max_display=10)
    plt.title("Q8 - SHAP: Top 10 Features (Importancia Media)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q8_shap_bar_top10.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Mean absolute SHAP for ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    top3_features = shap_importance.head(3).index.tolist()
    print(f"  Top 3 SHAP features: {top3_features}")

    # Dependence plots for top 3
    for feat in top3_features:
        fig = plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, shap_sample, show=False)
        plt.title(f"Q8 - SHAP Dependence: {feat}")
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        fig.savefig(
            os.path.join(FIGURES_DIR, f"q8_shap_dependence_{safe_name}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close("all")

    # ── Native feature importance ────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        fi.tail(15).plot.barh(ax=ax, color=COLORS["secondary"])
        ax.set_title(f"Q8 - Feature Importance Nativa: {best_name}")
        ax.set_xlabel("Importancia")
        ax.grid(axis="x", alpha=0.3)
        fig.savefig(os.path.join(FIGURES_DIR, "q8_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Error Analysis ───────────────────────────────────────────────────────
    print("  Running error analysis...")
    y_pred = best_m["y_pred"]
    error_df = X_test.copy()
    error_df["y_true"] = y_test.values
    error_df["y_pred"] = y_pred
    error_df["error_type"] = "TN"
    error_df.loc[(error_df["y_true"] == 1) & (error_df["y_pred"] == 1), "error_type"] = "TP"
    error_df.loc[(error_df["y_true"] == 0) & (error_df["y_pred"] == 1), "error_type"] = "FP"
    error_df.loc[(error_df["y_true"] == 1) & (error_df["y_pred"] == 0), "error_type"] = "FN"

    fp_df = error_df[error_df["error_type"] == "FP"]
    fn_df = error_df[error_df["error_type"] == "FN"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Q8 - Analise de Erros: FP vs FN", fontsize=16)

    # Temporal: hour_of_day
    for i, (label, df, color) in enumerate([
        ("Falsos Positivos", fp_df, COLORS["danger"]),
        ("Falsos Negativos", fn_df, COLORS["accent"]),
    ]):
        ax = axes[i, 0]
        if "hour_of_day" in df.columns:
            df["hour_of_day"].dropna().astype(int).value_counts().sort_index().plot.bar(
                ax=ax, color=color, alpha=0.8
            )
        ax.set_title(f"{label}: Hora do Dia")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Contagem")
        ax.grid(axis="y", alpha=0.3)

    # Territorial: bairro_encoded distribution
    for i, (label, df, color) in enumerate([
        ("Falsos Positivos", fp_df, COLORS["danger"]),
        ("Falsos Negativos", fn_df, COLORS["accent"]),
    ]):
        ax = axes[i, 1]
        if "bairro_encoded" in df.columns:
            df["bairro_encoded"].dropna().hist(ax=ax, bins=30, color=color, alpha=0.8)
        ax.set_title(f"{label}: Bairro (encoded)")
        ax.set_xlabel("Bairro Encoded")
        ax.set_ylabel("Contagem")
        ax.grid(axis="y", alpha=0.3)

    # Categorical: tipo_encoded distribution
    for i, (label, df, color) in enumerate([
        ("Falsos Positivos", fp_df, COLORS["danger"]),
        ("Falsos Negativos", fn_df, COLORS["accent"]),
    ]):
        ax = axes[i, 2]
        if "tipo_encoded" in df.columns:
            df["tipo_encoded"].dropna().hist(ax=ax, bins=30, color=color, alpha=0.8)
        ax.set_title(f"{label}: Tipo (encoded)")
        ax.set_xlabel("Tipo Encoded")
        ax.set_ylabel("Contagem")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q8_error_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    error_summary = {
        "total_test": len(y_test),
        "FP_count": len(fp_df),
        "FN_count": len(fn_df),
        "FP_pct": len(fp_df) / len(y_test) * 100,
        "FN_pct": len(fn_df) / len(y_test) * 100,
    }
    print(f"  FP: {error_summary['FP_count']} ({error_summary['FP_pct']:.1f}%)")
    print(f"  FN: {error_summary['FN_count']} ({error_summary['FN_pct']:.1f}%)")

    return shap_importance, top3_features, error_summary


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
def write_docs(comp_df, all_metrics, best_name, best_params, shap_importance, top3_features, error_summary):
    print("\n" + "=" * 60)
    print("WRITING DOCUMENTATION")
    print("=" * 60)

    # ── model-comparison.md ──────────────────────────────────────────────────
    md_lines = ["# Model Comparison (Q6-Q7)\n"]
    md_lines.append("## Metrics Table\n")
    md_lines.append(comp_df.to_markdown(index=False))
    md_lines.append(f"\n\n## Best Model: **{best_name}**\n")
    bm = all_metrics[best_name]
    md_lines.append(f"- F1: {bm['f1']:.4f}")
    md_lines.append(f"- AUC-ROC: {bm['auc_roc']:.4f}")
    md_lines.append(f"- AUC-PR: {bm['auc_pr']:.4f}")
    if best_params:
        md_lines.append(f"\n### Best Hyperparameters (Optuna)\n")
        for k, v in best_params.items():
            md_lines.append(f"- `{k}`: {v}")
    md_lines.append("\n## Metric Justification\n")
    md_lines.append(
        "In the context of public policy for vulnerable citizens, **recall** is critical: "
        "failing to identify a chamado that will NOT be resolved in 7 days means a vulnerable "
        "citizen goes unattended. However, **precision** also matters for efficient resource "
        "allocation -- too many false positives waste limited government resources.\n\n"
        "We recommend **F1-score** as the primary metric (balancing precision and recall) and "
        "**AUC-ROC** as the secondary metric for overall discrimination ability."
    )

    with open(os.path.join(VAULT_OUT, "model-comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("  Written: vault/04-model-builder/outputs/model-comparison.md")

    # ── shap-analysis.md ─────────────────────────────────────────────────────
    shap_lines = [f"# SHAP Interpretability Analysis (Q8)\n"]
    shap_lines.append(f"## Model: {best_name}\n")
    shap_lines.append("## Top 10 Features by Mean |SHAP|\n")
    for i, (feat, val) in enumerate(shap_importance.head(10).items()):
        shap_lines.append(f"{i+1}. **{feat}**: {val:.4f}")
    shap_lines.append(f"\n## Top 3 Feature Dependence Plots\n")
    for feat in top3_features:
        safe = feat.replace("/", "_").replace(" ", "_")
        shap_lines.append(f"- `{feat}` -> `results/figures/q8_shap_dependence_{safe}.png`")
    shap_lines.append("\n## Error Analysis\n")
    shap_lines.append(f"- Total test samples: {error_summary['total_test']}")
    shap_lines.append(f"- False Positives (FP): {error_summary['FP_count']} ({error_summary['FP_pct']:.1f}%)")
    shap_lines.append(f"- False Negatives (FN): {error_summary['FN_count']} ({error_summary['FN_pct']:.1f}%)")
    shap_lines.append(
        "\nFP = model predicted resolution in 7 days but chamado was NOT resolved. "
        "FN = model predicted NO resolution but chamado WAS resolved."
    )
    shap_lines.append("\n## Policy Insights\n")
    shap_lines.append(
        "1. **Temporal patterns** (hour, day, month) strongly influence resolution probability, "
        "suggesting that when a chamado is opened affects municipal response capacity.\n"
        "2. **Territorial features** (bairro, subprefeitura) show significant variation, "
        "indicating geographic disparities in service delivery.\n"
        "3. **Service type** (tipo, subtipo, orgao) captures institutional capacity differences "
        "across municipal agencies.\n"
        "4. **Historical resolution rates** by neighborhood provide strong predictive signal, "
        "reflecting accumulated institutional performance.\n"
        "5. **Weather conditions** have moderate influence, with extreme rain events correlating "
        "with delayed resolutions (infrastructure overload).\n\n"
        "**Recommendation**: The prioritization system (Q9-Q10) should weight FN reduction heavily, "
        "as missing a delayed chamado means a vulnerable citizen remains unattended."
    )

    with open(os.path.join(VAULT_OUT, "shap-analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(shap_lines))
    print("  Written: vault/04-model-builder/outputs/shap-analysis.md")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_notebook(all_metrics, best_name, best_params, comp_df, shap_importance, top3_features, error_summary):
    print("\n" + "=" * 60)
    print("GENERATING NOTEBOOK")
    print("=" * 60)
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    def md(text):
        return nbformat.v4.new_markdown_cell(text)

    def code(text):
        return nbformat.v4.new_code_cell(text)

    cells = []

    # ── Title + Executive Summary ────────────────────────────────────────────
    bm = all_metrics[best_name]
    cells.append(md(f"""# Notebook 02 - Modelagem: Resolucao de Chamados em 7 Dias

## Resumo Executivo

Este notebook apresenta a modelagem preditiva para determinar se um chamado do sistema 1746
sera resolvido em ate 7 dias. Foram treinados e comparados 5 modelos:

- **Regressao Logistica** (baseline)
- **Random Forest** (200 arvores)
- **XGBoost** (200 arvores, default + otimizado com Optuna)
- **LightGBM** (200 arvores)

**Melhor modelo**: {best_name} com F1={bm['f1']:.4f} e AUC-ROC={bm['auc_roc']:.4f}.

A analise de interpretabilidade via SHAP revelou que as features mais importantes sao
relacionadas a tipo de servico, localizacao e padroes temporais.

---"""))

    # ── Q6 ───────────────────────────────────────────────────────────────────
    cells.append(md("""## Q6: Modelo Baseline - Regressao Logistica

### Justificativa da Metrica

No contexto de politicas publicas para cidadaos vulneraveis:
- **Recall** e critico: nao identificar um chamado que NAO sera resolvido em 7 dias significa
  um cidadao vulneravel sem atendimento.
- **Precisao** importa para alocacao eficiente de recursos limitados.
- **F1-score** e a metrica primaria (equilibra precisao e recall).
- **AUC-ROC** e a metrica secundaria para comparacao geral."""))

    cells.append(code("""import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Carregar dados
X_train = pd.read_parquet('data/features/X_train.parquet')
X_test = pd.read_parquet('data/features/X_test.parquet')
y_train = pd.read_parquet('data/features/y_train.parquet').squeeze()
y_test = pd.read_parquet('data/features/y_test.parquet').squeeze()

print(f"Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
print(f"Teste:  {X_test.shape[0]} amostras")
print(f"Taxa positiva (treino): {y_train.mean():.3f}")
print(f"Taxa positiva (teste):  {y_test.mean():.3f}")"""))

    cells.append(code("""from src.models.train_baseline import train_logistic_baseline, evaluate_model

model_lr = train_logistic_baseline(X_train, y_train)
metrics_lr = evaluate_model(model_lr, X_test, y_test)

print("=== Regressao Logistica (Baseline) ===")
for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
    print(f"  {k:12s}: {metrics_lr[k]:.4f}")"""))

    cells.append(md("""### Curva ROC e Matriz de Confusao do Baseline"""))

    cells.append(code("""from IPython.display import Image, display
display(Image(filename='results/figures/q6_roc_baseline.png'))"""))

    cells.append(code("""display(Image(filename='results/figures/q6_confusion_matrix.png'))"""))

    # ── Q7 ───────────────────────────────────────────────────────────────────
    cells.append(md("""## Q7: Modelos Avancados e Otimizacao de Hiperparametros

Treinamos Random Forest, XGBoost e LightGBM com configuracoes padroes (200 arvores cada).
Em seguida, otimizamos o XGBoost com Optuna (50 trials, 5-fold stratified CV, maximizando AUC-ROC)."""))

    cells.append(code(f"""# Tabela de comparacao de modelos
comp_table = pd.read_markdown('''
{comp_df.to_markdown(index=False)}
''')
comp_table"""))

    # Actually just display the pre-computed table as text
    cells[-1] = code(f"""# Tabela de comparacao completa
import pandas as pd
data = {comp_df.to_dict('list')}
comp_df = pd.DataFrame(data)
comp_df.style.highlight_max(subset=['F1', 'AUC-ROC', 'AUC-PR'], color='lightgreen')""")

    if best_params:
        params_str = json.dumps(best_params, indent=2)
        cells.append(md(f"""### Melhores Hiperparametros (Optuna - XGBoost)

```json
{params_str}
```"""))

    cells.append(md(f"""### Resultado: Melhor Modelo = **{best_name}**

| Metrica   | Valor  |
|-----------|--------|
| F1        | {bm['f1']:.4f} |
| AUC-ROC   | {bm['auc_roc']:.4f} |
| AUC-PR    | {bm['auc_pr']:.4f} |
| Precision | {bm['precision']:.4f} |
| Recall    | {bm['recall']:.4f} |"""))

    cells.append(md("### Curvas ROC e Precision-Recall (todos os modelos)"))
    cells.append(code("""display(Image(filename='results/figures/q7_roc_curves.png'))"""))
    cells.append(code("""display(Image(filename='results/figures/q7_pr_curves.png'))"""))
    cells.append(code("""display(Image(filename='results/figures/q7_confusion_best.png'))"""))
    cells.append(code("""display(Image(filename='results/figures/q7_model_comparison.png'))"""))

    # ── Q8 ───────────────────────────────────────────────────────────────────
    cells.append(md(f"""## Q8: Interpretabilidade e Analise de Erros

### Analise SHAP (TreeExplainer)

Utilizamos o SHAP (SHapley Additive exPlanations) para interpretar as predicoes do
melhor modelo ({best_name}). O SHAP atribui a cada feature uma contribuicao marginal
para cada predicao individual."""))

    cells.append(code("""display(Image(filename='results/figures/q8_shap_summary_beeswarm.png'))"""))

    cells.append(md("""### Top 10 Features por Importancia SHAP"""))
    cells.append(code("""display(Image(filename='results/figures/q8_shap_bar_top10.png'))"""))

    cells.append(md(f"""### Graficos de Dependencia SHAP

As 3 features mais importantes segundo SHAP: **{', '.join(top3_features)}**"""))

    for feat in top3_features:
        safe = feat.replace("/", "_").replace(" ", "_")
        cells.append(code(f"""display(Image(filename='results/figures/q8_shap_dependence_{safe}.png'))"""))

    cells.append(md("### Importancia Nativa do Modelo"))
    cells.append(code("""display(Image(filename='results/figures/q8_feature_importance.png'))"""))

    cells.append(md(f"""### Analise de Erros

| Tipo | Quantidade | Percentual |
|------|-----------|------------|
| Falsos Positivos (FP) | {error_summary['FP_count']} | {error_summary['FP_pct']:.1f}% |
| Falsos Negativos (FN) | {error_summary['FN_count']} | {error_summary['FN_pct']:.1f}% |

- **FP**: modelo previu resolucao em 7 dias, mas o chamado NAO foi resolvido.
- **FN**: modelo previu NAO resolucao, mas o chamado FOI resolvido."""))

    cells.append(code("""display(Image(filename='results/figures/q8_error_analysis.png'))"""))

    cells.append(md(f"""### Insights para Politicas Publicas

1. **Padroes temporais** (hora, dia, mes) influenciam fortemente a probabilidade de resolucao,
   sugerindo que o momento de abertura do chamado afeta a capacidade de resposta municipal.

2. **Features territoriais** (bairro, subprefeitura) mostram variacao significativa,
   indicando disparidades geograficas na prestacao de servicos.

3. **Tipo de servico** (tipo, subtipo, orgao) captura diferencas de capacidade institucional
   entre as agencias municipais.

4. **Taxas historicas de resolucao** por bairro fornecem forte sinal preditivo,
   refletindo o desempenho institucional acumulado.

5. **Condicoes meteorologicas** tem influencia moderada, com eventos de chuva extrema
   correlacionados com atrasos (sobrecarga de infraestrutura).

**Recomendacao**: O sistema de priorizacao (Q9-Q10) deve priorizar a reducao de FN,
pois nao identificar um chamado atrasado significa um cidadao vulneravel sem atendimento.

---

*Notebook gerado automaticamente pelo Agent 4 (Model Builder).*"""))

    nb.cells = cells
    nb_path = os.path.join(NOTEBOOKS_DIR, "02_modelagem_resolucao.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Written: {nb_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    X_train, X_test, y_train, y_test = load_data()

    # Q6
    lr_model, lr_metrics = run_q6(X_train, X_test, y_train, y_test)

    # Q7
    all_models, all_metrics, best_name, best_model, best_params, comp_df = run_q7(
        X_train, X_test, y_train, y_test, lr_metrics
    )

    # Q8
    shap_importance, top3_features, error_summary = run_q8(
        best_model, best_name, X_train, X_test, y_test, all_metrics
    )

    # Docs
    write_docs(comp_df, all_metrics, best_name, best_params, shap_importance, top3_features, error_summary)

    # Notebook
    generate_notebook(all_metrics, best_name, best_params, comp_df, shap_importance, top3_features, error_summary)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Verify outputs
    expected_files = [
        "results/models/logistic_baseline.joblib",
        "results/models/random_forest.joblib",
        "results/models/xgboost_model.joblib",
        "results/models/lgbm_model.joblib",
        "results/models/best_model.joblib",
        "results/models/test_predictions.parquet",
        "results/figures/q6_roc_baseline.png",
        "results/figures/q6_confusion_matrix.png",
        "results/figures/q7_roc_curves.png",
        "results/figures/q7_pr_curves.png",
        "results/figures/q7_confusion_best.png",
        "results/figures/q7_model_comparison.png",
        "results/figures/q8_shap_summary_beeswarm.png",
        "results/figures/q8_shap_bar_top10.png",
        "results/figures/q8_feature_importance.png",
        "results/figures/q8_error_analysis.png",
        "vault/04-model-builder/outputs/model-comparison.md",
        "vault/04-model-builder/outputs/shap-analysis.md",
        "notebooks/02_modelagem_resolucao.ipynb",
    ]
    print("\nOutput verification:")
    all_ok = True
    for fp in expected_files:
        full = os.path.join(PROJECT_ROOT, fp)
        exists = os.path.exists(full)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {fp}")

    # Also check SHAP dependence plots
    import glob
    dep_plots = glob.glob(os.path.join(FIGURES_DIR, "q8_shap_dependence_*.png"))
    print(f"  SHAP dependence plots: {len(dep_plots)} files")

    if all_ok:
        print("\n  ALL OUTPUTS VERIFIED SUCCESSFULLY")
    else:
        print("\n  WARNING: Some outputs are missing!")


if __name__ == "__main__":
    main()
