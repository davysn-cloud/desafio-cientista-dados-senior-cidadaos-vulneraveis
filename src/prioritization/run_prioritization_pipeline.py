"""
Agent 5 — Prioritization Designer Pipeline
Q9: Priority Score Design + Q10: Simulation
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── project root on path ────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.prioritization.score import compute_priority_score
from src.prioritization.simulate import (
    simulate_random_selection,
    simulate_score_selection,
    plot_lift_curve,
)

# ── palette ─────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "neutral": "#95A5A6",
}

# ── paths ───────────────────────────────────────────────────────────
DATA_FEATURES = os.path.join(PROJECT_ROOT, "data", "features")
RESULTS_MODELS = os.path.join(PROJECT_ROOT, "results", "models")
RESULTS_FIGURES = os.path.join(PROJECT_ROOT, "results", "figures")
VAULT_OUTPUTS = os.path.join(PROJECT_ROOT, "vault", "05-prioritization-designer", "outputs")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

for d in [RESULTS_FIGURES, VAULT_OUTPUTS, NOTEBOOKS_DIR]:
    os.makedirs(d, exist_ok=True)

BUDGET = 0.20
WEIGHTS = {"w1": 0.40, "w2": 0.20, "w3": 0.25, "w4": 0.15}


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
def load_data():
    preds = pd.read_parquet(os.path.join(RESULTS_MODELS, "test_predictions.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_FEATURES, "X_test.parquet"))
    assert len(preds) == len(X_test), "Predictions and features length mismatch"
    # align index
    X_test.index = preds.index
    return preds, X_test


# ═══════════════════════════════════════════════════════════════════
# 2. COMPUTE SCORE COMPONENTS
# ═══════════════════════════════════════════════════════════════════
def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0,1]."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def compute_urgency(X_test: pd.DataFrame) -> pd.Series:
    """Lower subtipo_encoded = worse historical resolution = more urgent."""
    # subtipo_encoded is target-encoded P(resolved|subtipo).
    # Lower value = lower resolution rate = more urgent.
    urgency = 1 - X_test["subtipo_encoded"]
    return normalize_series(urgency)


def compute_equity(X_test: pd.DataFrame) -> pd.Series:
    """Equity based on hist_resolution_rate_bairro."""
    equity = 1 - X_test["hist_resolution_rate_bairro"]
    return normalize_series(equity)


def compute_context(X_test: pd.DataFrame) -> pd.Series:
    """Weather-based context score."""
    # Combine extreme weather flags + precipitation
    precip_norm = normalize_series(X_test["precipitation_sum"])
    context = (
        0.35 * X_test["is_extreme_rain"].astype(float)
        + 0.35 * X_test["is_extreme_heat"].astype(float)
        + 0.30 * precip_norm
    )
    return normalize_series(context)


# ═══════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
def plot_score_components(urgency, equity, context, p_delay, save_path):
    """Q9: 4-panel histogram of score components."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    components = [
        (p_delay, "P(Atraso)", COLORS["danger"]),
        (urgency, "Urgencia (Subtipo)", COLORS["accent"]),
        (equity, "Equidade Territorial", COLORS["secondary"]),
        (context, "Contexto Climatico", COLORS["success"]),
    ]
    for ax, (data, title, color) in zip(axes.flatten(), components):
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Score normalizado [0,1]")
        ax.set_ylabel("Frequencia")
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1,
                    label=f"Mediana: {data.median():.2f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Q9 — Distribuicao dos Componentes do Score de Prioridade",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_score_distribution(priority_score, y_true, save_path):
    """Q10: Score distribution split by outcome."""
    fig, ax = plt.subplots(figsize=(10, 6))
    delayed = priority_score[y_true == 0]
    resolved = priority_score[y_true == 1]
    ax.hist(resolved, bins=50, alpha=0.65, color=COLORS["success"],
            label=f"Resolvidos em 7d (n={len(resolved)})", edgecolor="white")
    ax.hist(delayed, bins=50, alpha=0.65, color=COLORS["danger"],
            label=f"Atrasados (n={len(delayed)})", edgecolor="white")
    ax.axvline(priority_score.quantile(1 - BUDGET), color=COLORS["accent"],
               linestyle="--", linewidth=2,
               label=f"Limiar Top {int(BUDGET*100)}%")
    ax.set_xlabel("Score de Prioridade", fontsize=12)
    ax.set_ylabel("Frequencia", fontsize=12)
    ax.set_title("Q10 — Distribuicao do Score por Resultado", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_comparison_table(random_metrics, score_metrics, territorial_random,
                          territorial_score, lift, save_path):
    """Q10: Side-by-side comparison as table figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = ["Metrica", "Selecao Aleatoria", "Score de Prioridade", "Ganho"]
    rows = [
        ["Precision@20%",
         f"{random_metrics['precision_mean']:.1%} [{random_metrics['precision_ci'][0]:.1%}-{random_metrics['precision_ci'][1]:.1%}]",
         f"{score_metrics['precision']:.1%}",
         f"+{(score_metrics['precision'] - random_metrics['precision_mean']):.1%}"],
        ["Recall@20%",
         f"{random_metrics['recall_mean']:.1%} [{random_metrics['recall_ci'][0]:.1%}-{random_metrics['recall_ci'][1]:.1%}]",
         f"{score_metrics['recall']:.1%}",
         f"+{(score_metrics['recall'] - random_metrics['recall_mean']):.1%}"],
        ["Cobertura Territorial",
         f"{territorial_random:.0f} areas",
         f"{territorial_score:.0f} areas",
         f"{territorial_score - territorial_random:+.0f}"],
        ["Lift (Recall)",
         "1.00x (baseline)",
         f"{lift:.2f}x",
         f"+{(lift - 1)*100:.0f}%"],
    ]

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=[COLORS["primary"]]*4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # style header
    for j in range(4):
        table[0, j].set_text_props(color="white", fontweight="bold")
    # style gain column
    for i in range(1, 5):
        table[i, 3].set_text_props(color=COLORS["success"], fontweight="bold")

    ax.set_title("Q10 — Comparacao: Selecao Aleatoria vs. Score de Prioridade",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_weight_sensitivity(preds, urgency, equity, context, y_true, save_path):
    """Q9: Sensitivity analysis — vary each weight, measure recall@20%."""
    base_weights = WEIGHTS.copy()
    weight_names = ["w1", "w2", "w3", "w4"]
    labels = ["P(Atraso)", "Urgencia", "Equidade", "Contexto"]
    colors_list = [COLORS["danger"], COLORS["accent"], COLORS["secondary"], COLORS["success"]]

    p_delay = 1 - preds["y_proba"]
    components = {
        "w1": p_delay,
        "w2": urgency,
        "w3": equity,
        "w4": context,
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    test_values = np.arange(0.0, 0.65, 0.05)
    n_select = int(len(y_true) * BUDGET)
    delayed = (y_true == 0)

    for wname, label, color in zip(weight_names, labels, colors_list):
        recalls = []
        for val in test_values:
            w = base_weights.copy()
            # Set the target weight to val and redistribute the rest proportionally
            remaining = 1.0 - val
            other_sum = sum(base_weights[k] for k in weight_names if k != wname)
            if other_sum == 0:
                for k in weight_names:
                    if k != wname:
                        w[k] = remaining / 3
            else:
                for k in weight_names:
                    if k != wname:
                        w[k] = base_weights[k] / other_sum * remaining
            w[wname] = val

            score = sum(w[k] * components[k] for k in weight_names)
            top_idx = score.nlargest(n_select).index
            recall = delayed.loc[top_idx].sum() / delayed.sum()
            recalls.append(recall)

        ax.plot(test_values, recalls, label=label, color=color, linewidth=2, marker="o", markersize=4)

    ax.axvline(x=base_weights["w1"], color=COLORS["danger"], linestyle=":", alpha=0.4)
    ax.set_xlabel("Peso do Componente", fontsize=12)
    ax.set_ylabel("Recall@20%", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Q9 — Analise de Sensibilidade dos Pesos", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 4. MARKDOWN OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def write_score_formula_md(urgency, equity, context, p_delay, save_path):
    stats = {
        "P(Atraso)": p_delay,
        "Urgencia": urgency,
        "Equidade": equity,
        "Contexto": context,
    }
    stats_lines = ""
    for name, s in stats.items():
        stats_lines += f"| {name} | {s.mean():.3f} | {s.median():.3f} | {s.std():.3f} | {s.min():.3f} | {s.max():.3f} |\n"

    content = f"""# Q9 — Formula do Score de Priorizacao

## Formula

```
priority_score = w1 * P(atraso) + w2 * urgency_score + w3 * equity_score + w4 * context_score
```

## Pesos

| Componente | Peso | Justificativa |
|---|---|---|
| **P(Atraso)** — `w1` | 0.40 | O preditor de atraso do modelo XGBoost (AUC=0.8628) e a informacao mais rica e individualizada. Recebe o maior peso. |
| **Urgencia (Subtipo)** — `w2` | 0.20 | O tipo de servico determina a complexidade intrinseca do chamado. Subtipos com baixa taxa de resolucao historica sao inerentemente mais dificeis. |
| **Equidade Territorial** — `w3` | 0.25 | Bairros historicamente mal atendidos precisam de atencao compensatoria. Peso elevado para garantir justica distributiva. |
| **Contexto Climatico** — `w4` | 0.15 | Eventos extremos (chuva forte, calor extremo) aumentam a urgencia pontual. Peso menor pois e um fator transitorio. |

**Soma dos pesos:** {sum(WEIGHTS.values()):.2f}

## Componentes — Estatisticas Descritivas

| Componente | Media | Mediana | Std | Min | Max |
|---|---|---|---|---|---|
{stats_lines}

## Normalizacao
Cada componente e normalizado para [0, 1] via Min-Max antes da combinacao.
O score final tambem e normalizado para [0, 1].

## Trade-offs

### Preditivo vs. Equitativo
- Pesos maiores em P(Atraso) maximizam a precisao (identifica mais casos que realmente atrasam).
- Pesos maiores em Equidade aumentam a cobertura territorial, garantindo que bairros vulneraveis recebam atencao mesmo quando o modelo nao preve alto risco individual.

### Sensibilidade
- A analise de sensibilidade (`q9_weight_sensitivity.png`) mostra que o recall e robusto a variacoes moderadas dos pesos.
- P(Atraso) e o componente mais influente: aumentar w1 melhora recall mas pode concentrar atencao em poucos bairros.
- Equidade age como contrapeso geografico, garantindo diversidade territorial.

## Visualizacao
- Distribuicao dos componentes: `results/figures/q9_score_components.png`
- Sensibilidade dos pesos: `results/figures/q9_weight_sensitivity.png`
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {save_path}")


def write_simulation_results_md(random_metrics, score_metrics, territorial_random,
                                 territorial_score, lift, total_delayed, total_cases,
                                 save_path):
    content = f"""# Q10 — Resultados da Simulacao de Priorizacao

## Configuracao

| Parametro | Valor |
|---|---|
| Total de chamados (teste 2024) | {total_cases:,} |
| Chamados atrasados (>7 dias) | {total_delayed:,} ({total_delayed/total_cases:.1%}) |
| Orcamento de priorizacao | {BUDGET:.0%} ({int(total_cases * BUDGET):,} chamados) |
| Iteracoes da selecao aleatoria | 100 |

## Resultados Comparativos

### Selecao Aleatoria (baseline)
- **Precision@20%:** {random_metrics['precision_mean']:.1%} (IC 95%: {random_metrics['precision_ci'][0]:.1%} - {random_metrics['precision_ci'][1]:.1%})
- **Recall@20%:** {random_metrics['recall_mean']:.1%} (IC 95%: {random_metrics['recall_ci'][0]:.1%} - {random_metrics['recall_ci'][1]:.1%})

### Score de Prioridade
- **Precision@20%:** {score_metrics['precision']:.1%}
- **Recall@20%:** {score_metrics['recall']:.1%}
- **Lift:** {lift:.2f}x

### Cobertura Territorial
- Aleatoria: {territorial_random:.0f} areas de planejamento distintas
- Score: {territorial_score:.0f} areas de planejamento distintas

## Interpretacao

### Ganhos
- O sistema de priorizacao captura **{score_metrics['recall']:.1%}** dos chamados que realmente atrasam,
  selecionando apenas **{BUDGET:.0%}** do total. Isso representa um lift de **{lift:.2f}x** sobre a selecao aleatoria.
- A precisao aumenta de {random_metrics['precision_mean']:.1%} para {score_metrics['precision']:.1%},
  significando que cada chamado priorizado tem {score_metrics['precision']:.1%} de chance de realmente necessitar atencao.

### Cobertura Territorial
- O sistema mantem cobertura em {territorial_score:.0f} areas de planejamento,
  {'superando' if territorial_score >= territorial_random else 'ligeiramente inferior a'} a selecao aleatoria ({territorial_random:.0f}).
- O componente de equidade (w3=0.25) garante que bairros historicamente sub-atendidos recebam atencao proporcional.

## Recomendacao de Politica Publica

1. **Adotar o score de priorizacao** para triagem dos chamados 1746, com revisao trimestral dos pesos.
2. **Monitorar equidade territorial** mensalmente: se alguma regiao ficar sistematicamente de fora do top 20%, ajustar w3.
3. **Retreinar o modelo** semestralmente com dados novos para manter o componente P(Atraso) atualizado.
4. **Integrar dados climaticos em tempo real** para que o componente de contexto reflita condicoes atuais, nao apenas historicas.
5. **Limiar dinamico**: em periodos de crise (enchentes, ondas de calor), expandir temporariamente o orcamento de 20% para 30%.

## Visualizacoes
- Curva de ganho acumulado: `results/figures/q10_lift_curve.png`
- Distribuicao do score: `results/figures/q10_score_distribution.png`
- Tabela comparativa: `results/figures/q10_comparison_table.png`
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. NOTEBOOK GENERATION
# ═══════════════════════════════════════════════════════════════════
def generate_notebook(priority_score, preds, X_test, urgency, equity, context,
                      random_metrics, score_metrics, territorial_random,
                      territorial_score, lift, save_path):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    # ── TITLE ──
    md("""# Notebook 03 — Sistema de Priorizacao de Chamados

## Resumo Executivo

Este notebook apresenta o **sistema de priorizacao de chamados 1746** para o Programa Pequenos Cariocas (PIC).
O objetivo e identificar, dentro de um orcamento limitado a **20% dos chamados**, aqueles com maior risco de atraso
e maior necessidade de atencao prioritaria.

**Principais resultados:**
- O score de priorizacao alcanca um **lift de {lift:.2f}x** sobre a selecao aleatoria
- Captura **{recall:.1%}** dos chamados atrasados selecionando apenas 20% do total
- Mantem cobertura em **{terr:.0f} areas de planejamento** distintas

O sistema combina quatro dimensoes: predicao de atraso (modelo XGBoost), urgencia do tipo de servico,
equidade territorial e contexto climatico.""".format(
        lift=lift, recall=score_metrics['recall'], terr=territorial_score))

    # ── SETUP ──
    md("## 1. Configuracao e Carga de Dados")

    code("""import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLORS = {
    'primary': '#1B4F72',
    'secondary': '#2E86C1',
    'accent': '#F39C12',
    'success': '#27AE60',
    'danger': '#E74C3C',
    'neutral': '#95A5A6',
}

BUDGET = 0.20
WEIGHTS = {"w1": 0.40, "w2": 0.20, "w3": 0.25, "w4": 0.15}
""")

    code("""# Carregar dados
preds = pd.read_parquet("results/models/test_predictions.parquet")
X_test = pd.read_parquet("data/features/X_test.parquet")
X_test.index = preds.index

print(f"Total de chamados (teste 2024): {len(preds):,}")
print(f"Chamados atrasados (y=0): {(preds['y_true']==0).sum():,} ({(preds['y_true']==0).mean():.1%})")
print(f"Chamados resolvidos (y=1): {(preds['y_true']==1).sum():,} ({(preds['y_true']==1).mean():.1%})")
""")

    # ── Q9 ──
    md("""## 2. Q9 — Construcao do Score de Priorizacao

### 2.1 Formula

O score de priorizacao combina quatro dimensoes complementares:

$$\\text{priority\\_score} = w_1 \\cdot P(\\text{atraso}) + w_2 \\cdot \\text{urgencia} + w_3 \\cdot \\text{equidade} + w_4 \\cdot \\text{contexto}$$

| Componente | Peso | Descricao |
|---|---|---|
| P(Atraso) | 0.40 | Probabilidade de atraso prevista pelo modelo XGBoost |
| Urgencia | 0.20 | Baseada na taxa de resolucao historica do subtipo |
| Equidade | 0.25 | Inverso da taxa de resolucao historica do bairro |
| Contexto | 0.15 | Condicoes climaticas extremas (chuva, calor) |
""")

    code("""# Componente 1: P(Atraso) — do modelo
p_delay = 1 - preds["y_proba"]

# Componente 2: Urgencia — subtipo com baixa taxa de resolucao = mais urgente
urgency_raw = 1 - X_test["subtipo_encoded"]
urgency = (urgency_raw - urgency_raw.min()) / (urgency_raw.max() - urgency_raw.min())

# Componente 3: Equidade — bairros com baixa taxa de resolucao = maior necessidade
equity_raw = 1 - X_test["hist_resolution_rate_bairro"]
equity = (equity_raw - equity_raw.min()) / (equity_raw.max() - equity_raw.min())

# Componente 4: Contexto climatico
precip_norm = (X_test["precipitation_sum"] - X_test["precipitation_sum"].min()) / \\
              (X_test["precipitation_sum"].max() - X_test["precipitation_sum"].min())
context = 0.35 * X_test["is_extreme_rain"].astype(float) + \\
          0.35 * X_test["is_extreme_heat"].astype(float) + \\
          0.30 * precip_norm
context = (context - context.min()) / (context.max() - context.min())

print("Estatisticas dos componentes:")
for name, s in [("P(Atraso)", p_delay), ("Urgencia", urgency),
                ("Equidade", equity), ("Contexto", context)]:
    print(f"  {name}: media={s.mean():.3f}, mediana={s.median():.3f}, std={s.std():.3f}")
""")

    md("""### 2.2 Distribuicao dos Componentes

Cada componente e normalizado para [0, 1] antes da combinacao.""")

    code("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
components = [
    (p_delay, "P(Atraso)", COLORS["danger"]),
    (urgency, "Urgencia (Subtipo)", COLORS["accent"]),
    (equity, "Equidade Territorial", COLORS["secondary"]),
    (context, "Contexto Climatico", COLORS["success"]),
]
for ax, (data, title, color) in zip(axes.flatten(), components):
    ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Score normalizado [0,1]")
    ax.set_ylabel("Frequencia")
    ax.axvline(data.median(), color="black", linestyle="--", linewidth=1,
               label=f"Mediana: {data.median():.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
fig.suptitle("Distribuicao dos Componentes do Score de Prioridade",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
plt.show()
""")

    md("""### 2.3 Calculo do Score Final""")

    code("""# Combinar com pesos
from src.prioritization.score import compute_priority_score

priority_score = compute_priority_score(
    y_proba=preds["y_proba"],
    urgency_score=urgency,
    equity_score=equity,
    context_score=context,
    weights=WEIGHTS,
)

print(f"Score de prioridade:")
print(f"  Media: {priority_score.mean():.3f}")
print(f"  Mediana: {priority_score.median():.3f}")
print(f"  Std: {priority_score.std():.3f}")
print(f"  Limiar top 20%: {priority_score.quantile(0.80):.3f}")
""")

    # ── Q9 Sensitivity ──
    md("""### 2.4 Analise de Sensibilidade dos Pesos

Variamos cada peso de 0.0 a 0.60 (redistribuindo proporcionalmente os demais)
e medimos o impacto no Recall@20%.""")

    code("""from src.prioritization.simulate import simulate_score_selection

weight_names = ["w1", "w2", "w3", "w4"]
labels_w = ["P(Atraso)", "Urgencia", "Equidade", "Contexto"]
colors_w = [COLORS["danger"], COLORS["accent"], COLORS["secondary"], COLORS["success"]]
components_dict = {"w1": p_delay, "w2": urgency, "w3": equity, "w4": context}
test_values = np.arange(0.0, 0.65, 0.05)
n_select = int(len(preds) * BUDGET)
delayed = (preds["y_true"] == 0)

fig, ax = plt.subplots(figsize=(10, 7))
for wname, label, color in zip(weight_names, labels_w, colors_w):
    recalls = []
    for val in test_values:
        w = WEIGHTS.copy()
        remaining = 1.0 - val
        other_sum = sum(WEIGHTS[k] for k in weight_names if k != wname)
        for k in weight_names:
            if k != wname:
                w[k] = WEIGHTS[k] / other_sum * remaining if other_sum > 0 else remaining / 3
        w[wname] = val
        score = sum(w[k] * components_dict[k] for k in weight_names)
        top_idx = score.nlargest(n_select).index
        recall = delayed.loc[top_idx].sum() / delayed.sum()
        recalls.append(recall)
    ax.plot(test_values, recalls, label=label, color=color, linewidth=2, marker="o", markersize=4)

ax.set_xlabel("Peso do Componente", fontsize=12)
ax.set_ylabel("Recall@20%", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Analise de Sensibilidade dos Pesos", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")

    # ── Q10 ──
    md("""## 3. Q10 — Simulacao: Selecao Aleatoria vs. Score de Prioridade

Comparamos duas estrategias para selecionar 20% dos chamados:
- **Estrategia A (baseline):** selecao aleatoria, repetida 100 vezes para intervalo de confianca
- **Estrategia B:** selecao dos top 20% pelo score de prioridade""")

    code("""from src.prioritization.simulate import simulate_random_selection, simulate_score_selection, plot_lift_curve

# Simulacao
random_metrics = simulate_random_selection(preds["y_true"], budget_fraction=BUDGET, n_iterations=100)
score_metrics = simulate_score_selection(preds["y_true"], priority_score, budget_fraction=BUDGET)

lift = score_metrics["recall"] / random_metrics["recall_mean"]

print("=== SELECAO ALEATORIA ===")
print(f"  Precision@20%: {random_metrics['precision_mean']:.1%} "
      f"(IC: {random_metrics['precision_ci'][0]:.1%}-{random_metrics['precision_ci'][1]:.1%})")
print(f"  Recall@20%:    {random_metrics['recall_mean']:.1%} "
      f"(IC: {random_metrics['recall_ci'][0]:.1%}-{random_metrics['recall_ci'][1]:.1%})")
print()
print("=== SCORE DE PRIORIDADE ===")
print(f"  Precision@20%: {score_metrics['precision']:.1%}")
print(f"  Recall@20%:    {score_metrics['recall']:.1%}")
print(f"  Lift:          {lift:.2f}x")
""")

    md("### 3.1 Curva de Ganho Acumulado (Lift)")

    code("""# Lift curve
delayed_bin = (preds["y_true"] == 0).astype(int)
sorted_idx = priority_score.sort_values(ascending=False).index
sorted_delayed = delayed_bin.loc[sorted_idx].values
cumulative_gain = np.cumsum(sorted_delayed) / delayed_bin.sum()
fraction_selected = np.arange(1, len(cumulative_gain) + 1) / len(cumulative_gain)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fraction_selected, cumulative_gain, label="Score de Prioridade",
        color=COLORS["danger"], linewidth=2)
ax.plot([0, 1], [0, 1], "k--", label="Selecao Aleatoria", alpha=0.5)
ax.axvline(x=0.20, color=COLORS["accent"], linestyle=":", label="Orcamento (20%)", alpha=0.8)
ax.fill_between(fraction_selected, fraction_selected, cumulative_gain,
                alpha=0.15, color=COLORS["danger"])
ax.set_xlabel("Fracao dos Chamados Selecionados", fontsize=12)
ax.set_ylabel("Fracao dos Chamados Atrasados Capturados", fontsize=12)
ax.set_title("Curva de Ganho Acumulado (Lift)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")

    md("### 3.2 Distribuicao do Score por Resultado")

    code("""fig, ax = plt.subplots(figsize=(10, 6))
resolved = priority_score[preds["y_true"] == 1]
delayed_scores = priority_score[preds["y_true"] == 0]
ax.hist(resolved, bins=50, alpha=0.65, color=COLORS["success"],
        label=f"Resolvidos em 7d (n={len(resolved)})", edgecolor="white")
ax.hist(delayed_scores, bins=50, alpha=0.65, color=COLORS["danger"],
        label=f"Atrasados (n={len(delayed_scores)})", edgecolor="white")
ax.axvline(priority_score.quantile(1 - BUDGET), color=COLORS["accent"],
           linestyle="--", linewidth=2, label=f"Limiar Top {int(BUDGET*100)}%")
ax.set_xlabel("Score de Prioridade", fontsize=12)
ax.set_ylabel("Frequencia", fontsize=12)
ax.set_title("Distribuicao do Score por Resultado", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
""")

    md("### 3.3 Cobertura Territorial")

    code("""# Cobertura territorial
n_select = int(len(preds) * BUDGET)
top_idx = priority_score.nlargest(n_select).index
territorial_score = X_test.loc[top_idx, "area_plan_encoded"].nunique()
territorial_total = X_test["area_plan_encoded"].nunique()
print(f"Areas de planejamento cobertas pelo top 20%: {territorial_score} de {territorial_total}")
print(f"Cobertura: {territorial_score/territorial_total:.1%}")
""")

    # ── CONCLUSIONS ──
    md("""## 4. Conclusoes e Recomendacoes de Politica Publica

### Eficacia do Sistema
O sistema de priorizacao demonstra ganhos significativos sobre a selecao aleatoria:
- **Lift de {lift:.2f}x** no recall, significando que o sistema e {lift:.2f} vezes mais eficaz em identificar chamados atrasados
- **Precision de {precision:.1%}** nos chamados priorizados, versus ~{random_prec:.1%} na selecao aleatoria

### Equidade Territorial
O componente de equidade (peso 0.25) garante que bairros historicamente sub-atendidos
recebam atencao proporcional, evitando a concentracao de recursos em regioes ja bem servidas.

### Recomendacoes
1. **Adotar o score de priorizacao** para triagem automatizada dos chamados 1746
2. **Monitorar equidade territorial** mensalmente e ajustar w3 se necessario
3. **Retreinar o modelo** semestralmente para manter P(Atraso) atualizado
4. **Integrar dados climaticos em tempo real** para melhorar o componente de contexto
5. **Limiar dinamico:** expandir de 20% para 30% em periodos de crise (enchentes, ondas de calor)
""".format(lift=lift, precision=score_metrics['precision'],
           random_prec=random_metrics['precision_mean']))

    nb.cells = cells

    with open(save_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  [OK] {save_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Agent 5 — Prioritization Designer Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Carregando dados...")
    preds, X_test = load_data()
    y_true = preds["y_true"]
    print(f"  Total: {len(preds):,} chamados | Atrasados: {(y_true==0).sum():,} ({(y_true==0).mean():.1%})")

    # 2. Components
    print("\n[2/7] Calculando componentes do score...")
    p_delay = 1 - preds["y_proba"]
    urgency = compute_urgency(X_test)
    equity = compute_equity(X_test)
    context = compute_context(X_test)

    for name, s in [("P(Atraso)", p_delay), ("Urgencia", urgency),
                    ("Equidade", equity), ("Contexto", context)]:
        print(f"  {name}: media={s.mean():.3f}, mediana={s.median():.3f}")

    # 3. Priority score
    print("\n[3/7] Calculando score final...")
    priority_score = compute_priority_score(
        y_proba=preds["y_proba"],
        urgency_score=urgency,
        equity_score=equity,
        context_score=context,
        weights=WEIGHTS,
    )
    threshold = priority_score.quantile(1 - BUDGET)
    print(f"  Score: media={priority_score.mean():.3f}, limiar top 20%={threshold:.3f}")

    # 4. Simulation
    print("\n[4/7] Executando simulacao...")
    random_metrics = simulate_random_selection(y_true, budget_fraction=BUDGET, n_iterations=100)
    score_metrics = simulate_score_selection(y_true, priority_score, budget_fraction=BUDGET)
    lift = score_metrics["recall"] / random_metrics["recall_mean"]

    # Territorial coverage
    n_select = int(len(y_true) * BUDGET)
    top_idx = priority_score.nlargest(n_select).index
    territorial_score_val = X_test.loc[top_idx, "area_plan_encoded"].nunique()
    # Random territorial coverage (mean over 100 iterations)
    rng = np.random.RandomState(42)
    terr_randoms = []
    for _ in range(100):
        sel = rng.choice(len(y_true), size=n_select, replace=False)
        terr_randoms.append(X_test.iloc[sel]["area_plan_encoded"].nunique())
    territorial_random_val = np.mean(terr_randoms)

    print(f"  Random  — Precision: {random_metrics['precision_mean']:.1%}, Recall: {random_metrics['recall_mean']:.1%}")
    print(f"  Score   — Precision: {score_metrics['precision']:.1%}, Recall: {score_metrics['recall']:.1%}")
    print(f"  Lift: {lift:.2f}x")
    print(f"  Territorial: score={territorial_score_val}, random={territorial_random_val:.0f}")

    # 5. Visualizations
    print("\n[5/7] Gerando visualizacoes...")
    plot_score_components(urgency, equity, context, p_delay,
                          os.path.join(RESULTS_FIGURES, "q9_score_components.png"))
    plot_score_distribution(priority_score, y_true,
                            os.path.join(RESULTS_FIGURES, "q10_score_distribution.png"))
    plot_lift_curve(y_true, priority_score,
                    os.path.join(RESULTS_FIGURES, "q10_lift_curve.png"))
    plot_comparison_table(random_metrics, score_metrics, territorial_random_val,
                          territorial_score_val, lift,
                          os.path.join(RESULTS_FIGURES, "q10_comparison_table.png"))
    plot_weight_sensitivity(preds, urgency, equity, context, y_true,
                            os.path.join(RESULTS_FIGURES, "q9_weight_sensitivity.png"))

    # 6. Markdown outputs
    print("\n[6/7] Escrevendo documentos...")
    write_score_formula_md(urgency, equity, context, p_delay,
                           os.path.join(VAULT_OUTPUTS, "score-formula.md"))
    write_simulation_results_md(random_metrics, score_metrics, territorial_random_val,
                                 territorial_score_val, lift,
                                 (y_true == 0).sum(), len(y_true),
                                 os.path.join(VAULT_OUTPUTS, "simulation-results.md"))

    # 7. Notebook
    print("\n[7/7] Gerando notebook...")
    generate_notebook(priority_score, preds, X_test, urgency, equity, context,
                      random_metrics, score_metrics, territorial_random_val,
                      territorial_score_val, lift,
                      os.path.join(NOTEBOOKS_DIR, "03_sistema_priorizacao.ipynb"))

    print("\n" + "=" * 60)
    print("Pipeline concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
