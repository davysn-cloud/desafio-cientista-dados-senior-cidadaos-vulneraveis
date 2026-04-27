import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_random_selection(
    y_true: pd.Series,
    budget_fraction: float = 0.20,
    n_iterations: int = 100,
    random_state: int = 42,
) -> dict:
    """Baseline: seleciona budget_fraction aleatoriamente, repete n_iterations vezes.
    Retorna média e IC 95% de precision e recall."""
    rng = np.random.RandomState(random_state)
    n_select = int(len(y_true) * budget_fraction)
    delayed = (y_true == 0)  # target=0 = não resolvido em 7 dias

    precisions, recalls = [], []
    for _ in range(n_iterations):
        selected_idx = rng.choice(len(y_true), size=n_select, replace=False)
        selected_delayed = delayed.iloc[selected_idx].sum()
        precisions.append(selected_delayed / n_select)
        recalls.append(selected_delayed / delayed.sum())

    return {
        "precision_mean": np.mean(precisions),
        "precision_ci": (np.percentile(precisions, 2.5), np.percentile(precisions, 97.5)),
        "recall_mean": np.mean(recalls),
        "recall_ci": (np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)),
    }


def simulate_score_selection(
    y_true: pd.Series,
    priority_score: pd.Series,
    budget_fraction: float = 0.20,
) -> dict:
    n_select = int(len(y_true) * budget_fraction)
    delayed = (y_true == 0)

    top_idx = priority_score.nlargest(n_select).index
    selected_delayed = delayed.loc[top_idx].sum()

    return {
        "precision": selected_delayed / n_select,
        "recall": selected_delayed / delayed.sum(),
    }


def plot_lift_curve(
    y_true: pd.Series,
    priority_score: pd.Series,
    save_path: str = "results/figures/q10_lift_curve.png",
):
    delayed = (y_true == 0).astype(int)
    sorted_idx = priority_score.sort_values(ascending=False).index
    sorted_delayed = delayed.loc[sorted_idx].values

    cumulative_gain = np.cumsum(sorted_delayed) / delayed.sum()
    fraction_selected = np.arange(1, len(cumulative_gain) + 1) / len(cumulative_gain)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fraction_selected, cumulative_gain, label="Score de Prioridade", color="#E74C3C", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Selecao Aleatoria", alpha=0.5)
    ax.axvline(x=0.20, color="#F39C12", linestyle=":", label="Orcamento (20%)", alpha=0.8)
    ax.set_xlabel("Fracao dos Chamados Selecionados")
    ax.set_ylabel("Fracao dos Chamados Atrasados Capturados")
    ax.set_title("Curva de Ganho Acumulado (Lift)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
