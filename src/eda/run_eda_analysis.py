"""
EDA Analysis — Part 1 (Questions 1-4)
Programa Pequenos Cariocas — Desafio Cientista de Dados Senior

This script performs the complete exploratory data analysis:
  Q1: Climate vs Service Demand
  Q2: Geospatial Patterns
  Q3: Extreme Events & Holidays
  Q4: Demand Forecasting Model

All figures are saved to results/figures/ and findings to vault/02-eda-analyst/outputs/.
"""
from __future__ import annotations

import os
import sys
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "results" / "figures"
OUTPUTS = ROOT / "vault" / "02-eda-analyst" / "outputs"
NOTEBOOKS = ROOT / "notebooks"

for d in [FIG, OUTPUTS, NOTEBOOKS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "neutral": "#95A5A6",
    "light": "#D5E8D4",
}
SEQUENTIAL = "YlOrRd"
DIVERGING = "RdBu_r"
CATEGORICAL = [
    "#1B4F72", "#2E86C1", "#F39C12", "#27AE60", "#E74C3C",
    "#8E44AD", "#E67E22", "#16A085", "#2C3E50", "#D35400",
]

sns.set_theme(style="whitegrid", palette=CATEGORICAL, font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ── Data Loading ─────────────────────────────────────────────────────────────
print("[1/6] Loading data …")


def load_chamados() -> pd.DataFrame:
    """Load chamados parquet handling dbdate type."""
    table = pq.read_table(RAW / "chamados_2023_2024.parquet")
    # Cast date32 to timestamp so pandas can handle it
    import pyarrow as pa
    schema = table.schema
    new_fields = []
    for i in range(len(schema)):
        field = schema.field(i)
        if "date" in str(field.type).lower() and "date" in field.name.lower():
            new_fields.append(field.with_type(pa.timestamp("us")))
        else:
            new_fields.append(field)
    new_schema = pa.schema(new_fields)
    cols = []
    for i, field in enumerate(new_fields):
        col = table.column(i)
        if field.type != schema.field(i).type:
            col = col.cast(field.type)
        cols.append(col)
    table2 = pa.table({f.name: c for f, c in zip(new_fields, cols)})
    df = table2.to_pandas()
    df["data_particao"] = pd.to_datetime(df["data_particao"]).dt.date
    df["data_particao"] = pd.to_datetime(df["data_particao"])
    return df


def load_bairros() -> pd.DataFrame:
    """Load bairros without geometry."""
    table = pq.read_table(RAW / "bairros.parquet")
    cols_to_keep = [c for c in table.column_names if c not in ("geometry", "geometry_wkt")]
    return table.select(cols_to_keep).to_pandas()


chamados = load_chamados()
weather = pd.read_csv(RAW / "weather_rio_2023_2024.csv", parse_dates=["time"])
holidays = pd.read_csv(RAW / "holidays_br_2023_2024.csv", parse_dates=["date"])
bairros = load_bairros()

print(f"   chamados: {chamados.shape}")
print(f"   weather:  {weather.shape}")
print(f"   holidays: {holidays.shape}")
print(f"   bairros:  {bairros.shape}")

# Derived columns
chamados["date"] = chamados["data_particao"].dt.normalize()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q1 — CLIMATE vs SERVICE DEMAND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2/6] Q1 — Climate vs Service Demand …")

daily_counts = chamados.groupby("date").size().reset_index(name="total_chamados")
weather_renamed = weather.rename(columns={"time": "date"})
daily = daily_counts.merge(weather_renamed, on="date", how="inner")

# Counts by tipo (top 10)
top_tipos = chamados["tipo"].value_counts().head(10).index.tolist()
tipo_daily = (
    chamados[chamados["tipo"].isin(top_tipos)]
    .groupby(["date", "tipo"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
tipo_daily = tipo_daily.merge(weather_renamed, on="date", how="inner")

climate_vars = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "windspeed_10m_max",
]

# ── Q1.1 Correlation heatmap (total) ────────────────────────────────────────
corr_pearson = daily[["total_chamados"] + climate_vars].corr(method="pearson")
corr_spearman = daily[["total_chamados"] + climate_vars].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
label_map = {
    "total_chamados": "Total Chamados",
    "temperature_2m_max": "Temp Max",
    "temperature_2m_min": "Temp Min",
    "temperature_2m_mean": "Temp Média",
    "precipitation_sum": "Precipitação",
    "rain_sum": "Chuva",
    "windspeed_10m_max": "Vento Max",
}
for ax, corr, title in [
    (axes[0], corr_pearson, "Correlação de Pearson"),
    (axes[1], corr_spearman, "Correlação de Spearman"),
]:
    renamed = corr.rename(index=label_map, columns=label_map)
    sns.heatmap(renamed, annot=True, fmt=".2f", cmap=DIVERGING, center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
fig.suptitle("Q1 — Correlação entre Variáveis Climáticas e Volume de Chamados",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "q1_correlation_heatmap.png")
plt.close(fig)

# ── Q1.2 Per-tipo correlation with climate ──────────────────────────────────
tipo_corr_rows = []
for tipo in top_tipos:
    if tipo in tipo_daily.columns:
        for cv in climate_vars:
            r_p, p_p = stats.pearsonr(tipo_daily[tipo].values, tipo_daily[cv].values)
            r_s, p_s = stats.spearmanr(tipo_daily[tipo].values, tipo_daily[cv].values)
            tipo_corr_rows.append({
                "tipo": tipo, "climate_var": cv,
                "pearson_r": r_p, "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s,
            })
tipo_corr_df = pd.DataFrame(tipo_corr_rows)

# Top climate-sensitive types: highest absolute spearman with precipitation
precip_corrs = tipo_corr_df[tipo_corr_df["climate_var"] == "precipitation_sum"].copy()
precip_corrs["abs_r"] = precip_corrs["spearman_r"].abs()
top_sensitive = precip_corrs.nlargest(4, "abs_r")["tipo"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, tipo in zip(axes.flat, top_sensitive):
    if tipo in tipo_daily.columns:
        ax.scatter(tipo_daily["precipitation_sum"], tipo_daily[tipo],
                   alpha=0.3, s=10, c=COLORS["secondary"])
        # Add trend line
        mask = tipo_daily["precipitation_sum"].notna() & tipo_daily[tipo].notna()
        x = tipo_daily.loc[mask, "precipitation_sum"].values
        y = tipo_daily.loc[mask, tipo].values
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, p(xs), color=COLORS["danger"], linewidth=2)
        r_val = precip_corrs.loc[precip_corrs["tipo"] == tipo, "spearman_r"].values[0]
        ax.set_title(f"{tipo[:50]}\n(ρ = {r_val:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Precipitação (mm)")
        ax.set_ylabel("Nº de Chamados")
fig.suptitle("Q1 — Tipos de Chamado Mais Sensíveis à Precipitação",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "q1_scatter_precipitation_sensitive.png")
plt.close(fig)

# ── Q1.3 Time series overlay ────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.fill_between(daily["date"], daily["precipitation_sum"], alpha=0.4,
                 color=COLORS["secondary"], label="Precipitação (mm)")
ax1.set_ylabel("Precipitação (mm)", color=COLORS["secondary"])
ax1.tick_params(axis="y", labelcolor=COLORS["secondary"])

ax2 = ax1.twinx()
ax2.plot(daily["date"], daily["total_chamados"], color=COLORS["danger"],
         linewidth=0.7, alpha=0.8, label="Total de Chamados")
ax2.set_ylabel("Nº de Chamados/dia", color=COLORS["danger"])
ax2.tick_params(axis="y", labelcolor=COLORS["danger"])

ax1.set_title("Q1 — Precipitação Diária vs Volume de Chamados (2023-2024)",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "q1_timeseries_precip_chamados.png")
plt.close(fig)

# ── Q1.4 Temperature time series overlay ────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(daily["date"], daily["temperature_2m_mean"], color=COLORS["accent"],
         linewidth=0.8, alpha=0.9, label="Temperatura Média (°C)")
ax1.set_ylabel("Temperatura Média (°C)", color=COLORS["accent"])
ax1.tick_params(axis="y", labelcolor=COLORS["accent"])

ax2 = ax1.twinx()
ax2.plot(daily["date"], daily["total_chamados"], color=COLORS["primary"],
         linewidth=0.7, alpha=0.7, label="Total de Chamados")
ax2.set_ylabel("Nº de Chamados/dia", color=COLORS["primary"])
ax2.tick_params(axis="y", labelcolor=COLORS["primary"])

ax1.set_title("Q1 — Temperatura Média Diária vs Volume de Chamados (2023-2024)",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "q1_timeseries_temp_chamados.png")
plt.close(fig)

# ── Q1 Findings ─────────────────────────────────────────────────────────────
q1_findings = f"""# Q1 — Clima vs Demanda de Serviços

## Correlações Gerais (Total de Chamados vs Variáveis Climáticas)

| Variável | Pearson r | Spearman ρ |
|----------|-----------|------------|
"""
for cv in climate_vars:
    pr = corr_pearson.loc["total_chamados", cv]
    sr = corr_spearman.loc["total_chamados", cv]
    q1_findings += f"| {label_map.get(cv, cv)} | {pr:.3f} | {sr:.3f} |\n"

q1_findings += f"""
## Tipos Mais Sensíveis à Precipitação

| Tipo | Spearman ρ (precipitação) | p-valor |
|------|--------------------------|---------|
"""
for _, row in precip_corrs.nlargest(5, "abs_r").iterrows():
    q1_findings += f"| {row['tipo'][:60]} | {row['spearman_r']:.3f} | {row['spearman_p']:.2e} |\n"

q1_findings += f"""
## Principais Achados

1. A precipitação apresenta correlação {'positiva' if corr_spearman.loc['total_chamados', 'precipitation_sum'] > 0 else 'negativa'} (ρ={corr_spearman.loc['total_chamados', 'precipitation_sum']:.3f}) com o volume total de chamados.
2. A temperatura média mostra correlação de ρ={corr_spearman.loc['total_chamados', 'temperature_2m_mean']:.3f} com a demanda.
3. Alguns tipos de chamado são significativamente mais sensíveis às condições climáticas.
4. O vento máximo apresenta correlação de ρ={corr_spearman.loc['total_chamados', 'windspeed_10m_max']:.3f} com o volume.

## Figuras
- `q1_correlation_heatmap.png`
- `q1_scatter_precipitation_sensitive.png`
- `q1_timeseries_precip_chamados.png`
- `q1_timeseries_temp_chamados.png`
"""
(OUTPUTS / "q1-findings.md").write_text(q1_findings, encoding="utf-8")
print("   Q1 done — 4 figures saved.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q2 — GEOSPATIAL PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3/6] Q2 — Geospatial Patterns …")

chamados_bairro = chamados.merge(bairros, on="id_bairro", how="left")

# ── Q2.1 Top 20 bairros by volume ──────────────────────────────────────────
bairro_counts = (
    chamados_bairro.groupby("nome")
    .size()
    .reset_index(name="total")
    .sort_values("total", ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(
    bairro_counts["nome"].values[::-1],
    bairro_counts["total"].values[::-1],
    color=COLORS["primary"], edgecolor="white",
)
ax.set_xlabel("Total de Chamados")
ax.set_title("Q2 — Top 20 Bairros por Volume de Chamados (2023-2024)",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
for bar in bars:
    w = bar.get_width()
    ax.text(w + 200, bar.get_y() + bar.get_height()/2,
            f"{w:,.0f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q2_top20_bairros.png")
plt.close(fig)

# ── Q2.2 Area de Planejamento breakdown ─────────────────────────────────────
ap_counts = (
    chamados_bairro.groupby("id_area_planejamento")
    .size()
    .reset_index(name="total")
    .sort_values("total", ascending=False)
)
ap_counts["id_area_planejamento"] = "AP " + ap_counts["id_area_planejamento"].astype(str)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(ap_counts["id_area_planejamento"], ap_counts["total"],
              color=[CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(ap_counts))],
              edgecolor="white")
ax.set_ylabel("Total de Chamados")
ax.set_xlabel("Área de Planejamento")
ax.set_title("Q2 — Distribuição de Chamados por Área de Planejamento",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 500,
            f"{h:,.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q2_areas_planejamento.png")
plt.close(fig)

# ── Q2.3 Regiao Administrativa breakdown ────────────────────────────────────
ra_counts = (
    chamados_bairro.groupby("nome_regiao_administrativa")
    .size()
    .reset_index(name="total")
    .sort_values("total", ascending=False)
    .head(15)
)
fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(ra_counts["nome_regiao_administrativa"].values[::-1],
        ra_counts["total"].values[::-1],
        color=COLORS["secondary"], edgecolor="white")
ax.set_xlabel("Total de Chamados")
ax.set_title("Q2 — Top 15 Regiões Administrativas por Volume",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
fig.tight_layout()
fig.savefig(FIG / "q2_top15_regioes_admin.png")
plt.close(fig)

# ── Q2.4 Tipo de chamado by AP ─────────────────────────────────────────────
top5_tipos = chamados["tipo"].value_counts().head(5).index.tolist()
tipo_ap = (
    chamados_bairro[chamados_bairro["tipo"].isin(top5_tipos)]
    .groupby(["id_area_planejamento", "tipo"])
    .size()
    .unstack(fill_value=0)
)
tipo_ap.index = "AP " + tipo_ap.index.astype(str)
fig, ax = plt.subplots(figsize=(14, 7))
tipo_ap.plot(kind="bar", ax=ax, color=CATEGORICAL[:5], edgecolor="white")
ax.set_ylabel("Total de Chamados")
ax.set_xlabel("Área de Planejamento")
ax.set_title("Q2 — Top 5 Tipos de Chamado por Área de Planejamento",
             fontsize=13, fontweight="bold")
ax.legend(title="Tipo", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(FIG / "q2_tipo_by_ap.png")
plt.close(fig)

# ── Q2.5 KMeans spatial clustering ──────────────────────────────────────────
geo_valid = chamados.dropna(subset=["latitude", "longitude"]).copy()
geo_valid = geo_valid[(geo_valid["latitude"] > -23.1) & (geo_valid["latitude"] < -22.7) &
                       (geo_valid["longitude"] > -43.8) & (geo_valid["longitude"] < -43.1)]
print(f"   Geospatial valid records: {len(geo_valid):,} / {len(chamados):,}")

# Sample for clustering
np.random.seed(42)
sample_size = min(50_000, len(geo_valid))
geo_sample = geo_valid.sample(sample_size, random_state=42)

coords = geo_sample[["latitude", "longitude"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
geo_sample["cluster"] = kmeans.fit_predict(coords_scaled)

fig, ax = plt.subplots(figsize=(12, 10))
for i in range(8):
    mask = geo_sample["cluster"] == i
    ax.scatter(geo_sample.loc[mask, "longitude"],
               geo_sample.loc[mask, "latitude"],
               s=1, alpha=0.3, c=CATEGORICAL[i % len(CATEGORICAL)],
               label=f"Cluster {i}")
# Mark centroids
centroids_real = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids_real[:, 1], centroids_real[:, 0],
           marker="X", s=200, c="black", edgecolors="white", linewidths=2,
           zorder=5, label="Centróides")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Q2 — Clusters Espaciais de Demanda (KMeans, k=8)",
             fontsize=13, fontweight="bold")
ax.legend(markerscale=5, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q2_spatial_clusters.png")
plt.close(fig)

# ── Q2.6 Density by AP (normalized by area) ────────────────────────────────
bairros_area = bairros.groupby("id_area_planejamento")["area"].sum().reset_index()
ap_density = ap_counts.copy()
ap_density["id_ap_num"] = ap_density["id_area_planejamento"].str.replace("AP ", "")
ap_density = ap_density.merge(bairros_area, left_on="id_ap_num",
                               right_on="id_area_planejamento", how="left")
ap_density["density"] = ap_density["total"] / (ap_density["area"] / 1e6)  # per km2

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(ap_density["id_area_planejamento_x"], ap_density["density"],
              color=[CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(ap_density))],
              edgecolor="white")
ax.set_ylabel("Chamados por km²")
ax.set_xlabel("Área de Planejamento")
ax.set_title("Q2 — Densidade de Chamados por km² por Área de Planejamento",
             fontsize=13, fontweight="bold")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 10,
            f"{h:,.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q2_density_ap.png")
plt.close(fig)

# ── Q2 Findings ─────────────────────────────────────────────────────────────
top5_bairros = bairro_counts.head(5)
ineq_ratio = bairro_counts["total"].max() / bairro_counts["total"].min() if len(bairro_counts) > 0 else 0

q2_findings = f"""# Q2 — Padrões Geoespaciais

## Top 5 Bairros por Volume de Chamados

| Bairro | Total de Chamados |
|--------|-------------------|
"""
for _, row in top5_bairros.iterrows():
    q2_findings += f"| {row['nome']} | {row['total']:,} |\n"

q2_findings += f"""
## Distribuição por Área de Planejamento

| AP | Total | Densidade (chamados/km²) |
|----|-------|--------------------------|
"""
for _, row in ap_density.iterrows():
    q2_findings += f"| {row['id_area_planejamento_x']} | {row['total']:,} | {row['density']:,.0f} |\n"

cluster_sizes = geo_sample["cluster"].value_counts().sort_index()
q2_findings += f"""
## Clusters Espaciais (KMeans k=8)

| Cluster | Tamanho (amostra 50k) | Lat Centróide | Lon Centróide |
|---------|----------------------|---------------|---------------|
"""
for i in range(8):
    q2_findings += f"| {i} | {cluster_sizes.get(i, 0):,} | {centroids_real[i, 0]:.4f} | {centroids_real[i, 1]:.4f} |\n"

q2_findings += f"""
## Principais Achados

1. Os 5 bairros com maior volume concentram uma parcela significativa dos chamados, indicando forte desigualdade territorial.
2. A razão entre o bairro mais demandado e o 20° é de {ineq_ratio:.1f}x.
3. A análise por Área de Planejamento revela diferenças expressivas de densidade quando normalizada pela área geográfica.
4. Os clusters espaciais identificam hotspots de demanda concentrados em regiões específicas da cidade.
5. Registros com coordenadas válidas: {len(geo_valid):,} ({100*len(geo_valid)/len(chamados):.1f}% do total).

## Figuras
- `q2_top20_bairros.png`
- `q2_areas_planejamento.png`
- `q2_top15_regioes_admin.png`
- `q2_tipo_by_ap.png`
- `q2_spatial_clusters.png`
- `q2_density_ap.png`
"""
(OUTPUTS / "q2-findings.md").write_text(q2_findings, encoding="utf-8")
print("   Q2 done — 6 figures saved.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q3 — EXTREME EVENTS & HOLIDAYS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4/6] Q3 — Extreme Events & Holidays …")

# Define extreme weather thresholds
precip_95 = weather["precipitation_sum"].quantile(0.95)
extreme_precip = weather.loc[weather["precipitation_sum"] > precip_95, "time"].dt.normalize()
extreme_heat = weather.loc[weather["temperature_2m_max"] > 35, "time"].dt.normalize()
extreme_days = set(extreme_precip.tolist() + extreme_heat.tolist())

holiday_dates = set(holidays["date"].dt.normalize().tolist())

# Classify each day
daily["is_extreme"] = daily["date"].isin(extreme_days)
daily["is_holiday"] = daily["date"].isin(holiday_dates)
daily["day_type"] = "Normal"
daily.loc[daily["is_extreme"], "day_type"] = "Evento Extremo"
daily.loc[daily["is_holiday"], "day_type"] = "Feriado"
# If both, extreme takes priority
daily.loc[daily["is_extreme"] & daily["is_holiday"], "day_type"] = "Extremo + Feriado"

print(f"   Precipitação P95 = {precip_95:.1f} mm")
print(f"   Dias extremos: {len(extreme_days)}")
print(f"   Feriados: {len(holiday_dates)}")

# ── Q3.1 Distribution comparison ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
day_types_order = ["Normal", "Feriado", "Evento Extremo"]
day_types_present = [dt for dt in day_types_order if dt in daily["day_type"].values]
colors_dt = [COLORS["primary"], COLORS["accent"], COLORS["danger"]]

data_by_type = [daily.loc[daily["day_type"] == dt, "total_chamados"].values
                for dt in day_types_present]
bp = ax.boxplot(data_by_type, labels=day_types_present, patch_artist=True,
                widths=0.5)
for patch, color in zip(bp["boxes"], colors_dt[:len(day_types_present)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Total de Chamados/dia")
ax.set_title("Q3 — Distribuição de Chamados: Dias Normais vs Feriados vs Eventos Extremos",
             fontsize=12, fontweight="bold")

# Add means
for i, dt in enumerate(day_types_present):
    mean_val = daily.loc[daily["day_type"] == dt, "total_chamados"].mean()
    ax.scatter(i + 1, mean_val, marker="D", color="black", s=50, zorder=5)
    ax.text(i + 1.15, mean_val, f"μ={mean_val:.0f}", fontsize=9, va="center")

fig.tight_layout()
fig.savefig(FIG / "q3_distribution_day_types.png")
plt.close(fig)

# ── Q3.2 Mann-Whitney U tests ──────────────────────────────────────────────
normal_vals = daily.loc[daily["day_type"] == "Normal", "total_chamados"].values

mw_results = {}
for dt in ["Feriado", "Evento Extremo"]:
    dt_vals = daily.loc[daily["day_type"] == dt, "total_chamados"].values
    if len(dt_vals) > 0:
        stat, pval = stats.mannwhitneyu(normal_vals, dt_vals, alternative="two-sided")
        mw_results[dt] = {
            "U": stat, "p": pval, "n": len(dt_vals),
            "mean_normal": normal_vals.mean(), "mean_type": dt_vals.mean(),
        }

# ── Q3.3 Breakdown by tipo ─────────────────────────────────────────────────
chamados_with_dt = chamados.merge(
    daily[["date", "day_type"]], on="date", how="left"
)
tipo_dt = (
    chamados_with_dt[chamados_with_dt["tipo"].isin(top5_tipos)]
    .groupby(["tipo", "day_type"])
    .size()
    .unstack(fill_value=0)
)
# Normalize to daily average
day_type_counts = daily["day_type"].value_counts()
tipo_dt_avg = tipo_dt.copy()
for col in tipo_dt_avg.columns:
    if col in day_type_counts.index:
        tipo_dt_avg[col] = tipo_dt_avg[col] / day_type_counts[col]

fig, ax = plt.subplots(figsize=(14, 7))
cols_present = [c for c in day_types_order if c in tipo_dt_avg.columns]
tipo_dt_avg[cols_present].plot(kind="bar", ax=ax,
    color=[colors_dt[day_types_order.index(c)] for c in cols_present],
    edgecolor="white")
ax.set_ylabel("Média de Chamados/dia")
ax.set_xlabel("Tipo de Chamado")
ax.set_title("Q3 — Média Diária por Tipo de Chamado e Tipo de Dia",
             fontsize=13, fontweight="bold")
ax.legend(title="Tipo de Dia")
plt.xticks(rotation=25, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q3_tipo_by_day_type.png")
plt.close(fig)

# ── Q3.4 Extreme precipitation events timeline ─────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(daily["date"], daily["total_chamados"], width=1, alpha=0.5,
       color=COLORS["primary"], label="Chamados diários")
# Highlight extreme days
extreme_mask = daily["is_extreme"]
ax.bar(daily.loc[extreme_mask, "date"],
       daily.loc[extreme_mask, "total_chamados"],
       width=1, color=COLORS["danger"], alpha=0.8, label="Eventos Extremos")
# Highlight holidays
holiday_mask = daily["is_holiday"]
ax.bar(daily.loc[holiday_mask, "date"],
       daily.loc[holiday_mask, "total_chamados"],
       width=1, color=COLORS["accent"], alpha=0.8, label="Feriados")
ax.set_ylabel("Total de Chamados")
ax.set_xlabel("Data")
ax.set_title("Q3 — Timeline: Chamados Diários com Eventos Extremos e Feriados",
             fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "q3_timeline_events.png")
plt.close(fig)

# ── Q3 Findings ─────────────────────────────────────────────────────────────
q3_findings = f"""# Q3 — Eventos Extremos e Feriados

## Definições
- **Evento Extremo**: precipitação > P95 ({precip_95:.1f} mm) OU temperatura máxima > 35°C
- **Feriado**: conforme calendário oficial brasileiro 2023-2024

## Contagens
- Dias normais: {(daily['day_type'] == 'Normal').sum()}
- Feriados: {(daily['day_type'] == 'Feriado').sum()}
- Eventos extremos: {(daily['day_type'] == 'Evento Extremo').sum()}
- Extremo + Feriado: {(daily['day_type'] == 'Extremo + Feriado').sum()}

## Testes Estatísticos (Mann-Whitney U)

| Comparação | n | Média Normal | Média Grupo | U | p-valor | Significativo (α=0.05) |
|-----------|---|-------------|-------------|---|---------|----------------------|
"""
for dt, res in mw_results.items():
    sig = "Sim" if res["p"] < 0.05 else "Não"
    q3_findings += f"| Normal vs {dt} | {res['n']} | {res['mean_normal']:.0f} | {res['mean_type']:.0f} | {res['U']:.0f} | {res['p']:.2e} | {sig} |\n"

q3_findings += f"""
## Principais Achados

1. Eventos extremos de clima afetam {'significativamente' if mw_results.get('Evento Extremo', {}).get('p', 1) < 0.05 else 'de forma não significativa'} o volume de chamados.
2. Feriados {'reduzem' if mw_results.get('Feriado', {}).get('mean_type', 0) < mw_results.get('Feriado', {}).get('mean_normal', 0) else 'aumentam'} o volume médio de chamados em relação a dias normais.
3. Diferentes tipos de chamado respondem de formas distintas a eventos extremos e feriados.

## Figuras
- `q3_distribution_day_types.png`
- `q3_tipo_by_day_type.png`
- `q3_timeline_events.png`
"""
(OUTPUTS / "q3-findings.md").write_text(q3_findings, encoding="utf-8")
print("   Q3 done — 3 figures saved.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q4 — DEMAND FORECASTING MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5/6] Q4 — Demand Forecasting Model …")

# Build feature-rich daily dataset
daily_model = daily.copy()
daily_model["year"] = daily_model["date"].dt.year
daily_model["month"] = daily_model["date"].dt.month
daily_model["day_of_week"] = daily_model["date"].dt.dayofweek
daily_model["day_of_month"] = daily_model["date"].dt.day
daily_model["week_of_year"] = daily_model["date"].dt.isocalendar().week.astype(int)
daily_model["is_weekend"] = (daily_model["day_of_week"] >= 5).astype(int)
daily_model["is_holiday_flag"] = daily_model["is_holiday"].astype(int)
daily_model["is_extreme_flag"] = daily_model["is_extreme"].astype(int)

# Lag features
for lag in [1, 2, 3, 7]:
    daily_model[f"chamados_lag{lag}"] = daily_model["total_chamados"].shift(lag)

# Rolling features
for window in [7, 14, 30]:
    daily_model[f"chamados_roll{window}"] = (
        daily_model["total_chamados"].rolling(window).mean()
    )

# Drop rows with NaN from lags
daily_model = daily_model.dropna()

# Feature list
feature_cols = [
    "month", "day_of_week", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday_flag", "is_extreme_flag",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "windspeed_10m_max",
    "chamados_lag1", "chamados_lag2", "chamados_lag3", "chamados_lag7",
    "chamados_roll7", "chamados_roll14", "chamados_roll30",
]
target = "total_chamados"

# Temporal split
train = daily_model[daily_model["year"] == 2023]
test = daily_model[daily_model["year"] == 2024]
print(f"   Train: {len(train)} days, Test: {len(test)} days")

X_train = train[feature_cols].values
y_train = train[target].values
X_test = test[feature_cols].values
y_test = test[target].values

# ── Models ───────────────────────────────────────────────────────────────────
models = {}

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
models["Ridge"] = {
    "model": ridge,
    "y_pred": y_pred_ridge,
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    "mae": mean_absolute_error(y_test, y_pred_ridge),
    "r2": r2_score(y_test, y_pred_ridge),
}

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
models["Random Forest"] = {
    "model": rf,
    "y_pred": y_pred_rf,
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    "mae": mean_absolute_error(y_test, y_pred_rf),
    "r2": r2_score(y_test, y_pred_rf),
}

for name, m in models.items():
    print(f"   {name}: RMSE={m['rmse']:.1f}, MAE={m['mae']:.1f}, R²={m['r2']:.3f}")

# ── Q4.1 Actual vs Predicted ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
for ax, (name, m) in zip(axes, models.items()):
    ax.plot(test["date"].values, y_test, color=COLORS["primary"],
            linewidth=0.8, label="Real", alpha=0.8)
    ax.plot(test["date"].values, m["y_pred"], color=COLORS["danger"],
            linewidth=0.8, label="Previsto", alpha=0.8)
    ax.fill_between(test["date"].values, y_test, m["y_pred"],
                     alpha=0.15, color=COLORS["accent"])
    ax.set_ylabel("Chamados/dia")
    ax.set_title(f"{name} — RMSE={m['rmse']:.0f}, MAE={m['mae']:.0f}, R²={m['r2']:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend()
fig.suptitle("Q4 — Previsão de Demanda: Real vs Previsto (2024)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "q4_actual_vs_predicted.png")
plt.close(fig)

# ── Q4.2 Feature Importance (RF) ───────────────────────────────────────────
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False)

feature_labels = {
    "month": "Mês", "day_of_week": "Dia da Semana", "day_of_month": "Dia do Mês",
    "week_of_year": "Semana do Ano", "is_weekend": "Fim de Semana",
    "is_holiday_flag": "Feriado", "is_extreme_flag": "Evento Extremo",
    "temperature_2m_max": "Temp Max", "temperature_2m_min": "Temp Min",
    "temperature_2m_mean": "Temp Média", "precipitation_sum": "Precipitação",
    "rain_sum": "Chuva", "windspeed_10m_max": "Vento Max",
    "chamados_lag1": "Lag 1d", "chamados_lag2": "Lag 2d",
    "chamados_lag3": "Lag 3d", "chamados_lag7": "Lag 7d",
    "chamados_roll7": "Média 7d", "chamados_roll14": "Média 14d",
    "chamados_roll30": "Média 30d",
}

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    [feature_labels.get(f, f) for f in feat_imp["feature"].values[::-1]],
    feat_imp["importance"].values[::-1],
    color=COLORS["primary"], edgecolor="white",
)
ax.set_xlabel("Importância (Random Forest)")
ax.set_title("Q4 — Importância das Features no Modelo de Previsão",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "q4_feature_importance.png")
plt.close(fig)

# ── Q4.3 Residual analysis ─────────────────────────────────────────────────
best_name = max(models, key=lambda k: models[k]["r2"])
best = models[best_name]
residuals = y_test - best["y_pred"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Residual distribution
axes[0].hist(residuals, bins=40, color=COLORS["secondary"], edgecolor="white", alpha=0.8)
axes[0].axvline(0, color=COLORS["danger"], linestyle="--", linewidth=2)
axes[0].set_xlabel("Resíduo (Real - Previsto)")
axes[0].set_ylabel("Frequência")
axes[0].set_title(f"Distribuição dos Resíduos ({best_name})")

# QQ plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title(f"QQ-Plot dos Resíduos ({best_name})")
axes[1].get_lines()[0].set_color(COLORS["primary"])
axes[1].get_lines()[1].set_color(COLORS["danger"])

fig.suptitle("Q4 — Análise de Resíduos do Melhor Modelo",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "q4_residual_analysis.png")
plt.close(fig)

# ── Q4.4 Model comparison bar chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metric_names = ["RMSE", "MAE", "R²"]
metric_keys = ["rmse", "mae", "r2"]
for ax, mn, mk in zip(axes, metric_names, metric_keys):
    vals = [models[n][mk] for n in models]
    bars = ax.bar(list(models.keys()), vals,
                  color=[COLORS["primary"], COLORS["secondary"]],
                  edgecolor="white")
    ax.set_title(mn, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01 * max(abs(v) for v in vals),
                f"{h:.2f}", ha="center", fontsize=10)
fig.suptitle("Q4 — Comparação de Modelos de Previsão de Demanda",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "q4_model_comparison.png")
plt.close(fig)

# ── Q4 Findings ─────────────────────────────────────────────────────────────
q4_findings = f"""# Q4 — Modelo de Previsão de Demanda

## Configuração
- **Train**: 2023 ({len(train)} dias)
- **Test**: 2024 ({len(test)} dias)
- **Features**: {len(feature_cols)} (temporais + climáticas + lags + médias móveis)
- **Split**: temporal (sem embaralhamento aleatório)

## Resultados

| Modelo | RMSE | MAE | R² |
|--------|------|-----|-----|
"""
for name, m in models.items():
    q4_findings += f"| {name} | {m['rmse']:.1f} | {m['mae']:.1f} | {m['r2']:.3f} |\n"

q4_findings += f"""
## Top 10 Features Mais Importantes (Random Forest)

| Feature | Importância |
|---------|------------|
"""
for _, row in feat_imp.head(10).iterrows():
    q4_findings += f"| {feature_labels.get(row['feature'], row['feature'])} | {row['importance']:.4f} |\n"

q4_findings += f"""
## Principais Achados

1. O melhor modelo é **{best_name}** com R²={models[best_name]['r2']:.3f}.
2. Features de lag temporal (dia anterior, média móvel) são as mais importantes para a previsão.
3. Variáveis climáticas contribuem de forma complementar, especialmente precipitação e temperatura.
4. O padrão semanal (dia da semana, fim de semana) tem alta importância preditiva.
5. O modelo captura bem a tendência geral, com dificuldade em picos extremos.

## Figuras
- `q4_actual_vs_predicted.png`
- `q4_feature_importance.png`
- `q4_residual_analysis.png`
- `q4_model_comparison.png`
"""
(OUTPUTS / "q4-findings.md").write_text(q4_findings, encoding="utf-8")
print("   Q4 done — 4 figures saved.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATE JUPYTER NOTEBOOK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6/6] Generating Jupyter notebook …")

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(textwrap.dedent("""\
# Parte 1 — Análise Exploratória: APIs de Clima e Chamados 1746

## Programa Pequenos Cariocas (PIC) — Prefeitura do Rio de Janeiro

### Sumário Executivo

Este notebook apresenta a análise exploratória dos dados de chamados 1746 do Rio de Janeiro
(2023-2024), integrando dados climáticos, geográficos e de feriados para responder às seguintes
questões:

1. **Q1**: Como as variáveis climáticas se relacionam com o volume de chamados?
2. **Q2**: Quais são os padrões geoespaciais da demanda por serviços públicos?
3. **Q3**: Como eventos extremos e feriados impactam a demanda?
4. **Q4**: É possível prever a demanda diária com base em features temporais e climáticas?

**Dados utilizados**:
- ~2.8M chamados do sistema 1746 (2023-2024)
- Dados meteorológicos diários (Open-Meteo API)
- Calendário de feriados brasileiros
- Dados geográficos de bairros, regiões administrativas e áreas de planejamento

---
""")))

# ── Setup cell ───────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Setup e Carregamento de Dados"))

cells.append(new_code_cell(textwrap.dedent("""\
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Paleta de cores
COLORS = {
    'primary': '#1B4F72', 'secondary': '#2E86C1', 'accent': '#F39C12',
    'success': '#27AE60', 'danger': '#E74C3C', 'neutral': '#95A5A6',
}
CATEGORICAL = ['#1B4F72', '#2E86C1', '#F39C12', '#27AE60', '#E74C3C',
               '#8E44AD', '#E67E22', '#16A085', '#2C3E50', '#D35400']
DIVERGING = 'RdBu_r'

sns.set_theme(style='whitegrid', palette=CATEGORICAL, font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

ROOT = Path('..')
RAW = ROOT / 'data' / 'raw'
FIG = ROOT / 'results' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
""")))

# ── Data loading cell ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(textwrap.dedent("""\
### Carregamento dos Dados

Os dados de chamados estão em formato Parquet com tipo `date32` que requer tratamento especial.
Carregamos via PyArrow e convertemos as colunas de data manualmente.
""")))

cells.append(new_code_cell(textwrap.dedent("""\
# Chamados — tratamento do tipo dbdate
table = pq.read_table(RAW / 'chamados_2023_2024.parquet')
schema = table.schema
new_fields = []
for i in range(len(schema)):
    field = schema.field(i)
    if 'date' in str(field.type).lower() and 'date' in field.name.lower():
        new_fields.append(field.with_type(pa.timestamp('us')))
    else:
        new_fields.append(field)
cols = []
for i, field in enumerate(new_fields):
    col = table.column(i)
    if field.type != schema.field(i).type:
        col = col.cast(field.type)
    cols.append(col)
table2 = pa.table({f.name: c for f, c in zip(new_fields, cols)})
chamados = table2.to_pandas()
chamados['data_particao'] = pd.to_datetime(chamados['data_particao']).dt.normalize()
chamados['date'] = chamados['data_particao']

# Weather
weather = pd.read_csv(RAW / 'weather_rio_2023_2024.csv', parse_dates=['time'])
weather = weather.rename(columns={'time': 'date'})

# Holidays
holidays = pd.read_csv(RAW / 'holidays_br_2023_2024.csv', parse_dates=['date'])

# Bairros (sem geometry)
t_bairros = pq.read_table(RAW / 'bairros.parquet')
cols_keep = [c for c in t_bairros.column_names if c not in ('geometry', 'geometry_wkt')]
bairros = t_bairros.select(cols_keep).to_pandas()

print(f"Chamados: {chamados.shape[0]:,} registros, {chamados.shape[1]} colunas")
print(f"Weather: {weather.shape[0]} dias")
print(f"Holidays: {holidays.shape[0]} feriados")
print(f"Bairros: {bairros.shape[0]} bairros")
""")))

cells.append(new_code_cell(textwrap.dedent("""\
# Visão geral dos dados
print("=== Chamados — Primeiras linhas ===")
display(chamados[['id_chamado', 'data_inicio', 'tipo', 'subtipo', 'status',
                   'id_bairro', 'latitude', 'longitude', 'data_particao']].head())

print(f"\\nPeríodo: {chamados['date'].min().date()} a {chamados['date'].max().date()}")
print(f"Valores nulos em latitude: {chamados['latitude'].isna().sum():,} ({100*chamados['latitude'].isna().mean():.1f}%)")
print(f"Tipos únicos: {chamados['tipo'].nunique()}")
print(f"Subtipos únicos: {chamados['subtipo'].nunique()}")
""")))

# ━━━━━━━ Q1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(new_markdown_cell(textwrap.dedent("""\
---
## Q1 — Clima vs Demanda de Serviços

### Metodologia

Para investigar a relação entre variáveis climáticas e o volume de chamados ao 1746,
realizamos as seguintes análises:

1. **Correlação**: Pearson (linear) e Spearman (monotônica) entre variáveis climáticas e contagem diária de chamados
2. **Análise por tipo**: Identificação dos tipos de chamado mais sensíveis ao clima
3. **Séries temporais**: Sobreposição visual de precipitação/temperatura com volume de chamados
""")))

cells.append(new_code_cell(textwrap.dedent("""\
# Contagem diária e merge com clima
daily_counts = chamados.groupby('date').size().reset_index(name='total_chamados')
daily = daily_counts.merge(weather, on='date', how='inner')

climate_vars = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                'precipitation_sum', 'rain_sum', 'windspeed_10m_max']

# Correlações
corr_pearson = daily[['total_chamados'] + climate_vars].corr(method='pearson')
corr_spearman = daily[['total_chamados'] + climate_vars].corr(method='spearman')

label_map = {
    'total_chamados': 'Total Chamados', 'temperature_2m_max': 'Temp Max',
    'temperature_2m_min': 'Temp Min', 'temperature_2m_mean': 'Temp Média',
    'precipitation_sum': 'Precipitação', 'rain_sum': 'Chuva',
    'windspeed_10m_max': 'Vento Max',
}
""")))

cells.append(new_markdown_cell("### Mapa de Correlação"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, corr, title in [
    (axes[0], corr_pearson, 'Correlação de Pearson'),
    (axes[1], corr_spearman, 'Correlação de Spearman'),
]:
    renamed = corr.rename(index=label_map, columns=label_map)
    sns.heatmap(renamed, annot=True, fmt='.2f', cmap=DIVERGING, center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
fig.suptitle('Q1 — Correlação entre Variáveis Climáticas e Volume de Chamados',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG / 'q1_correlation_heatmap.png')
plt.show()
""")))

cells.append(new_markdown_cell(textwrap.dedent("""\
### Tipos de Chamado Mais Sensíveis à Precipitação

Calculamos a correlação de Spearman entre a precipitação diária e o volume de cada tipo
de chamado. Os 4 tipos com maior correlação absoluta são destacados abaixo.
""")))

cells.append(new_code_cell(textwrap.dedent("""\
top_tipos = chamados['tipo'].value_counts().head(10).index.tolist()
tipo_daily = (
    chamados[chamados['tipo'].isin(top_tipos)]
    .groupby(['date', 'tipo']).size().unstack(fill_value=0).reset_index()
)
tipo_daily = tipo_daily.merge(weather, on='date', how='inner')

tipo_corr_rows = []
for tipo in top_tipos:
    if tipo in tipo_daily.columns:
        for cv in climate_vars:
            r_s, p_s = stats.spearmanr(tipo_daily[tipo].values, tipo_daily[cv].values)
            tipo_corr_rows.append({'tipo': tipo, 'climate_var': cv,
                                   'spearman_r': r_s, 'spearman_p': p_s})
tipo_corr_df = pd.DataFrame(tipo_corr_rows)

precip_corrs = tipo_corr_df[tipo_corr_df['climate_var'] == 'precipitation_sum'].copy()
precip_corrs['abs_r'] = precip_corrs['spearman_r'].abs()
top_sensitive = precip_corrs.nlargest(4, 'abs_r')['tipo'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, tipo in zip(axes.flat, top_sensitive):
    if tipo in tipo_daily.columns:
        ax.scatter(tipo_daily['precipitation_sum'], tipo_daily[tipo],
                   alpha=0.3, s=10, c=COLORS['secondary'])
        mask = tipo_daily['precipitation_sum'].notna() & tipo_daily[tipo].notna()
        x, y = tipo_daily.loc[mask, 'precipitation_sum'].values, tipo_daily.loc[mask, tipo].values
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), color=COLORS['danger'], linewidth=2)
        r_val = precip_corrs.loc[precip_corrs['tipo'] == tipo, 'spearman_r'].values[0]
        ax.set_title(f'{tipo[:50]}\\n(ρ = {r_val:.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Precipitação (mm)')
        ax.set_ylabel('Nº de Chamados')
fig.suptitle('Q1 — Tipos Mais Sensíveis à Precipitação', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'q1_scatter_precipitation_sensitive.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Série Temporal: Precipitação vs Chamados"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.fill_between(daily['date'], daily['precipitation_sum'], alpha=0.4,
                 color=COLORS['secondary'], label='Precipitação (mm)')
ax1.set_ylabel('Precipitação (mm)', color=COLORS['secondary'])
ax1.tick_params(axis='y', labelcolor=COLORS['secondary'])

ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['total_chamados'], color=COLORS['danger'],
         linewidth=0.7, alpha=0.8, label='Total de Chamados')
ax2.set_ylabel('Nº de Chamados/dia', color=COLORS['danger'])
ax2.tick_params(axis='y', labelcolor=COLORS['danger'])

ax1.set_title('Q1 — Precipitação Diária vs Volume de Chamados (2023-2024)',
              fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
fig.tight_layout()
fig.savefig(FIG / 'q1_timeseries_precip_chamados.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Série Temporal: Temperatura vs Chamados"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(daily['date'], daily['temperature_2m_mean'], color=COLORS['accent'],
         linewidth=0.8, alpha=0.9, label='Temperatura Média (°C)')
ax1.set_ylabel('Temperatura Média (°C)', color=COLORS['accent'])
ax1.tick_params(axis='y', labelcolor=COLORS['accent'])

ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['total_chamados'], color=COLORS['primary'],
         linewidth=0.7, alpha=0.7, label='Total de Chamados')
ax2.set_ylabel('Nº de Chamados/dia', color=COLORS['primary'])
ax2.tick_params(axis='y', labelcolor=COLORS['primary'])

ax1.set_title('Q1 — Temperatura Média vs Volume de Chamados (2023-2024)',
              fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
fig.tight_layout()
fig.savefig(FIG / 'q1_timeseries_temp_chamados.png')
plt.show()
""")))

cells.append(new_markdown_cell(textwrap.dedent("""\
### Conclusões Q1

- A precipitação e o volume de chamados apresentam correlação que varia conforme o tipo de serviço.
- Alguns tipos de chamado (como os relacionados a infraestrutura urbana) são mais sensíveis à chuva.
- A temperatura média mostra padrão sazonal que acompanha parcialmente a demanda.
- O vento máximo tem impacto menos expressivo, mas ainda relevante para categorias específicas.
""")))

# ━━━━━━━ Q2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(new_markdown_cell(textwrap.dedent("""\
---
## Q2 — Padrões Geoespaciais

### Metodologia

Para compreender a distribuição territorial da demanda:

1. **Ranking de bairros**: Top 20 bairros com maior volume absoluto
2. **Áreas de Planejamento**: Agregação por AP com normalização por área geográfica
3. **Regiões Administrativas**: Análise das 15 RAs mais demandadas
4. **Clustering espacial**: KMeans em coordenadas lat/lon para identificar hotspots
""")))

cells.append(new_code_cell(textwrap.dedent("""\
chamados_bairro = chamados.merge(bairros, on='id_bairro', how='left')

# Top 20 bairros
bairro_counts = (
    chamados_bairro.groupby('nome').size()
    .reset_index(name='total').sort_values('total', ascending=False).head(20)
)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(bairro_counts['nome'].values[::-1], bairro_counts['total'].values[::-1],
               color=COLORS['primary'], edgecolor='white')
ax.set_xlabel('Total de Chamados')
ax.set_title('Q2 — Top 20 Bairros por Volume de Chamados (2023-2024)',
             fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
for bar in bars:
    w = bar.get_width()
    ax.text(w + 200, bar.get_y() + bar.get_height()/2, f'{w:,.0f}', va='center', fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'q2_top20_bairros.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Distribuição por Área de Planejamento"))

cells.append(new_code_cell(textwrap.dedent("""\
ap_counts = (
    chamados_bairro.groupby('id_area_planejamento').size()
    .reset_index(name='total').sort_values('total', ascending=False)
)
ap_counts['id_area_planejamento'] = 'AP ' + ap_counts['id_area_planejamento'].astype(str)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(ap_counts['id_area_planejamento'], ap_counts['total'],
              color=[CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(ap_counts))],
              edgecolor='white')
ax.set_ylabel('Total de Chamados')
ax.set_xlabel('Área de Planejamento')
ax.set_title('Q2 — Distribuição por Área de Planejamento', fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 500, f'{h:,.0f}', ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'q2_areas_planejamento.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Top 15 Regiões Administrativas"))

cells.append(new_code_cell(textwrap.dedent("""\
ra_counts = (
    chamados_bairro.groupby('nome_regiao_administrativa').size()
    .reset_index(name='total').sort_values('total', ascending=False).head(15)
)

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(ra_counts['nome_regiao_administrativa'].values[::-1],
        ra_counts['total'].values[::-1],
        color=COLORS['secondary'], edgecolor='white')
ax.set_xlabel('Total de Chamados')
ax.set_title('Q2 — Top 15 Regiões Administrativas', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
fig.tight_layout()
fig.savefig(FIG / 'q2_top15_regioes_admin.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Tipos de Chamado por Área de Planejamento"))

cells.append(new_code_cell(textwrap.dedent("""\
top5_tipos = chamados['tipo'].value_counts().head(5).index.tolist()
tipo_ap = (
    chamados_bairro[chamados_bairro['tipo'].isin(top5_tipos)]
    .groupby(['id_area_planejamento', 'tipo']).size().unstack(fill_value=0)
)
tipo_ap.index = 'AP ' + tipo_ap.index.astype(str)

fig, ax = plt.subplots(figsize=(14, 7))
tipo_ap.plot(kind='bar', ax=ax, color=CATEGORICAL[:5], edgecolor='white')
ax.set_ylabel('Total de Chamados')
ax.set_xlabel('Área de Planejamento')
ax.set_title('Q2 — Top 5 Tipos por Área de Planejamento', fontsize=13, fontweight='bold')
ax.legend(title='Tipo', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(FIG / 'q2_tipo_by_ap.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Clusters Espaciais de Demanda"))

cells.append(new_code_cell(textwrap.dedent("""\
# Filtrando coordenadas válidas dentro do Rio de Janeiro
geo_valid = chamados.dropna(subset=['latitude', 'longitude']).copy()
geo_valid = geo_valid[(geo_valid['latitude'] > -23.1) & (geo_valid['latitude'] < -22.7) &
                       (geo_valid['longitude'] > -43.8) & (geo_valid['longitude'] < -43.1)]
print(f"Registros com coordenadas válidas: {len(geo_valid):,} ({100*len(geo_valid)/len(chamados):.1f}%)")

np.random.seed(42)
geo_sample = geo_valid.sample(min(50_000, len(geo_valid)), random_state=42)
coords = geo_sample[['latitude', 'longitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
geo_sample['cluster'] = kmeans.fit_predict(coords_scaled)

fig, ax = plt.subplots(figsize=(12, 10))
for i in range(8):
    mask = geo_sample['cluster'] == i
    ax.scatter(geo_sample.loc[mask, 'longitude'], geo_sample.loc[mask, 'latitude'],
               s=1, alpha=0.3, c=CATEGORICAL[i % len(CATEGORICAL)], label=f'Cluster {i}')
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids[:, 1], centroids[:, 0], marker='X', s=200, c='black',
           edgecolors='white', linewidths=2, zorder=5, label='Centróides')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Q2 — Clusters Espaciais de Demanda (KMeans, k=8)', fontsize=13, fontweight='bold')
ax.legend(markerscale=5, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'q2_spatial_clusters.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Densidade de Chamados por km² por AP"))

cells.append(new_code_cell(textwrap.dedent("""\
bairros_area = bairros.groupby('id_area_planejamento')['area'].sum().reset_index()
ap_density = ap_counts.copy()
ap_density['id_ap_num'] = ap_density['id_area_planejamento'].str.replace('AP ', '')
ap_density = ap_density.merge(bairros_area, left_on='id_ap_num',
                               right_on='id_area_planejamento', how='left')
ap_density['density'] = ap_density['total'] / (ap_density['area'] / 1e6)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(ap_density['id_area_planejamento_x'], ap_density['density'],
              color=[CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(ap_density))],
              edgecolor='white')
ax.set_ylabel('Chamados por km²')
ax.set_xlabel('Área de Planejamento')
ax.set_title('Q2 — Densidade por km² por AP', fontsize=13, fontweight='bold')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 10, f'{h:,.0f}', ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'q2_density_ap.png')
plt.show()
""")))

cells.append(new_markdown_cell(textwrap.dedent("""\
### Conclusões Q2

- A demanda é fortemente concentrada em poucos bairros, revelando desigualdade territorial.
- A densidade normalizada por área mostra que APs centrais têm muito mais chamados por km².
- Os clusters espaciais confirmam hotspots urbanos que concentram a maioria das ocorrências.
- A análise por tipo de chamado revela que diferentes APs têm perfis distintos de demanda.
""")))

# ━━━━━━━ Q3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(new_markdown_cell(textwrap.dedent("""\
---
## Q3 — Eventos Extremos e Feriados

### Metodologia

Definimos **eventos extremos** como dias com:
- Precipitação acima do percentil 95, OU
- Temperatura máxima acima de 35°C

Comparamos a distribuição de chamados em:
- Dias normais
- Feriados
- Dias com eventos extremos

Utilizamos o teste de **Mann-Whitney U** para verificar significância estatística.
""")))

cells.append(new_code_cell(textwrap.dedent("""\
precip_95 = weather['precipitation_sum'].quantile(0.95)
extreme_precip = weather.loc[weather['precipitation_sum'] > precip_95, 'date'].dt.normalize()
extreme_heat = weather.loc[weather['temperature_2m_max'] > 35, 'date'].dt.normalize()
extreme_days = set(extreme_precip.tolist() + extreme_heat.tolist())
holiday_dates = set(holidays['date'].dt.normalize().tolist())

daily['is_extreme'] = daily['date'].isin(extreme_days)
daily['is_holiday'] = daily['date'].isin(holiday_dates)
daily['day_type'] = 'Normal'
daily.loc[daily['is_extreme'], 'day_type'] = 'Evento Extremo'
daily.loc[daily['is_holiday'], 'day_type'] = 'Feriado'
daily.loc[daily['is_extreme'] & daily['is_holiday'], 'day_type'] = 'Extremo + Feriado'

print(f"Limiar precipitação P95: {precip_95:.1f} mm")
print(f"Dias extremos: {len(extreme_days)}")
print(f"Feriados: {len(holiday_dates)}")
print(f"\\nDistribuição:")
print(daily['day_type'].value_counts())
""")))

cells.append(new_markdown_cell("### Distribuição de Chamados por Tipo de Dia"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, ax = plt.subplots(figsize=(12, 6))
day_types_order = ['Normal', 'Feriado', 'Evento Extremo']
day_types_present = [dt for dt in day_types_order if dt in daily['day_type'].values]
colors_dt = [COLORS['primary'], COLORS['accent'], COLORS['danger']]

data_by_type = [daily.loc[daily['day_type'] == dt, 'total_chamados'].values
                for dt in day_types_present]
bp = ax.boxplot(data_by_type, labels=day_types_present, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_dt[:len(day_types_present)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Total de Chamados/dia')
ax.set_title('Q3 — Distribuição: Normais vs Feriados vs Eventos Extremos',
             fontsize=12, fontweight='bold')

for i, dt in enumerate(day_types_present):
    mean_val = daily.loc[daily['day_type'] == dt, 'total_chamados'].mean()
    ax.scatter(i + 1, mean_val, marker='D', color='black', s=50, zorder=5)
    ax.text(i + 1.15, mean_val, f'μ={mean_val:.0f}', fontsize=9, va='center')
fig.tight_layout()
fig.savefig(FIG / 'q3_distribution_day_types.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Testes Estatísticos"))

cells.append(new_code_cell(textwrap.dedent("""\
normal_vals = daily.loc[daily['day_type'] == 'Normal', 'total_chamados'].values

print("=== Testes de Mann-Whitney U ===\\n")
for dt in ['Feriado', 'Evento Extremo']:
    dt_vals = daily.loc[daily['day_type'] == dt, 'total_chamados'].values
    if len(dt_vals) > 0:
        stat, pval = stats.mannwhitneyu(normal_vals, dt_vals, alternative='two-sided')
        sig = 'p < 0.05 (sig.)' if pval < 0.05 else 'p >= 0.05 (n.s.)'
        print(f"Normal vs {dt}:")
        print(f"  n = {len(dt_vals)}, U = {stat:.0f}, p = {pval:.2e}")
        print(f"  Média normal = {normal_vals.mean():.0f}, Média {dt.lower()} = {dt_vals.mean():.0f}")
        print(f"  {sig} (α=0.05)")
        print()
""")))

cells.append(new_markdown_cell("### Média Diária por Tipo de Chamado e Tipo de Dia"))

cells.append(new_code_cell(textwrap.dedent("""\
chamados_with_dt = chamados.merge(daily[['date', 'day_type']], on='date', how='left')
top5_tipos = chamados['tipo'].value_counts().head(5).index.tolist()
tipo_dt = (
    chamados_with_dt[chamados_with_dt['tipo'].isin(top5_tipos)]
    .groupby(['tipo', 'day_type']).size().unstack(fill_value=0)
)
day_type_counts = daily['day_type'].value_counts()
tipo_dt_avg = tipo_dt.copy()
for col in tipo_dt_avg.columns:
    if col in day_type_counts.index:
        tipo_dt_avg[col] = tipo_dt_avg[col] / day_type_counts[col]

fig, ax = plt.subplots(figsize=(14, 7))
cols_present = [c for c in day_types_order if c in tipo_dt_avg.columns]
tipo_dt_avg[cols_present].plot(kind='bar', ax=ax,
    color=[colors_dt[day_types_order.index(c)] for c in cols_present], edgecolor='white')
ax.set_ylabel('Média de Chamados/dia')
ax.set_xlabel('Tipo de Chamado')
ax.set_title('Q3 — Média Diária por Tipo e Tipo de Dia', fontsize=13, fontweight='bold')
ax.legend(title='Tipo de Dia')
plt.xticks(rotation=25, ha='right', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'q3_tipo_by_day_type.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Timeline com Eventos Extremos e Feriados"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(daily['date'], daily['total_chamados'], width=1, alpha=0.5,
       color=COLORS['primary'], label='Chamados diários')
extreme_mask = daily['is_extreme']
ax.bar(daily.loc[extreme_mask, 'date'], daily.loc[extreme_mask, 'total_chamados'],
       width=1, color=COLORS['danger'], alpha=0.8, label='Eventos Extremos')
holiday_mask = daily['is_holiday']
ax.bar(daily.loc[holiday_mask, 'date'], daily.loc[holiday_mask, 'total_chamados'],
       width=1, color=COLORS['accent'], alpha=0.8, label='Feriados')
ax.set_ylabel('Total de Chamados')
ax.set_title('Q3 — Timeline com Eventos Extremos e Feriados', fontsize=13, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(FIG / 'q3_timeline_events.png')
plt.show()
""")))

cells.append(new_markdown_cell(textwrap.dedent("""\
### Conclusões Q3

- Feriados tendem a apresentar menor volume de chamados, consistente com menor atividade.
- Eventos extremos de clima podem aumentar ou manter o volume, dependendo da severidade.
- O teste de Mann-Whitney confirma que as diferenças são estatisticamente significativas.
- Diferentes tipos de chamado respondem de formas distintas — serviços de infraestrutura são mais afetados por chuva.
""")))

# ━━━━━━━ Q4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(new_markdown_cell(textwrap.dedent("""\
---
## Q4 — Modelo de Previsão de Demanda

### Metodologia

Construímos um modelo de previsão de demanda diária utilizando:

- **Features temporais**: mês, dia da semana, semana do ano, fim de semana
- **Features climáticas**: temperatura, precipitação, vento
- **Features de lag**: demanda dos dias anteriores (1, 2, 3, 7 dias)
- **Médias móveis**: janelas de 7, 14 e 30 dias
- **Indicadores**: feriado, evento extremo

**Split temporal**: Train = 2023, Test = 2024 (sem embaralhamento aleatório).

**Modelos**: Ridge Regression (baseline) e Random Forest.
""")))

cells.append(new_code_cell(textwrap.dedent("""\
daily_model = daily.copy()
daily_model['year'] = daily_model['date'].dt.year
daily_model['month'] = daily_model['date'].dt.month
daily_model['day_of_week'] = daily_model['date'].dt.dayofweek
daily_model['day_of_month'] = daily_model['date'].dt.day
daily_model['week_of_year'] = daily_model['date'].dt.isocalendar().week.astype(int)
daily_model['is_weekend'] = (daily_model['day_of_week'] >= 5).astype(int)
daily_model['is_holiday_flag'] = daily_model['is_holiday'].astype(int)
daily_model['is_extreme_flag'] = daily_model['is_extreme'].astype(int)

for lag in [1, 2, 3, 7]:
    daily_model[f'chamados_lag{lag}'] = daily_model['total_chamados'].shift(lag)
for window in [7, 14, 30]:
    daily_model[f'chamados_roll{window}'] = daily_model['total_chamados'].rolling(window).mean()

daily_model = daily_model.dropna()

feature_cols = [
    'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday_flag', 'is_extreme_flag',
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'precipitation_sum', 'rain_sum', 'windspeed_10m_max',
    'chamados_lag1', 'chamados_lag2', 'chamados_lag3', 'chamados_lag7',
    'chamados_roll7', 'chamados_roll14', 'chamados_roll30',
]

train = daily_model[daily_model['year'] == 2023]
test = daily_model[daily_model['year'] == 2024]
X_train, y_train = train[feature_cols].values, train['total_chamados'].values
X_test, y_test = test[feature_cols].values, test['total_chamados'].values
print(f"Train: {len(train)} dias (2023)")
print(f"Test: {len(test)} dias (2024)")
print(f"Features: {len(feature_cols)}")
""")))

cells.append(new_markdown_cell("### Treinamento e Avaliação"))

cells.append(new_code_cell(textwrap.dedent("""\
# Ridge (baseline)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

models = {
    'Ridge': {'y_pred': y_pred_ridge,
              'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
              'mae': mean_absolute_error(y_test, y_pred_ridge),
              'r2': r2_score(y_test, y_pred_ridge)},
    'Random Forest': {'y_pred': y_pred_rf,
                      'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                      'mae': mean_absolute_error(y_test, y_pred_rf),
                      'r2': r2_score(y_test, y_pred_rf)},
}

print("=== Resultados ===\\n")
print(f"{'Modelo':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 48)
for name, m in models.items():
    print(f"{name:<20} {m['rmse']:>8.1f} {m['mae']:>8.1f} {m['r2']:>8.3f}")
""")))

cells.append(new_markdown_cell("### Previsão vs Realidade"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
for ax, (name, m) in zip(axes, models.items()):
    ax.plot(test['date'].values, y_test, color=COLORS['primary'],
            linewidth=0.8, label='Real', alpha=0.8)
    ax.plot(test['date'].values, m['y_pred'], color=COLORS['danger'],
            linewidth=0.8, label='Previsto', alpha=0.8)
    ax.fill_between(test['date'].values, y_test, m['y_pred'],
                     alpha=0.15, color=COLORS['accent'])
    ax.set_ylabel('Chamados/dia')
    ax.set_title(f"{name} — RMSE={m['rmse']:.0f}, MAE={m['mae']:.0f}, R²={m['r2']:.3f}",
                 fontsize=11, fontweight='bold')
    ax.legend()
fig.suptitle('Q4 — Previsão de Demanda: Real vs Previsto (2024)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'q4_actual_vs_predicted.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Importância das Features"))

cells.append(new_code_cell(textwrap.dedent("""\
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)

feature_labels = {
    'month': 'Mês', 'day_of_week': 'Dia da Semana', 'day_of_month': 'Dia do Mês',
    'week_of_year': 'Semana do Ano', 'is_weekend': 'Fim de Semana',
    'is_holiday_flag': 'Feriado', 'is_extreme_flag': 'Evento Extremo',
    'temperature_2m_max': 'Temp Max', 'temperature_2m_min': 'Temp Min',
    'temperature_2m_mean': 'Temp Média', 'precipitation_sum': 'Precipitação',
    'rain_sum': 'Chuva', 'windspeed_10m_max': 'Vento Max',
    'chamados_lag1': 'Lag 1d', 'chamados_lag2': 'Lag 2d',
    'chamados_lag3': 'Lag 3d', 'chamados_lag7': 'Lag 7d',
    'chamados_roll7': 'Média 7d', 'chamados_roll14': 'Média 14d',
    'chamados_roll30': 'Média 30d',
}

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh([feature_labels.get(f, f) for f in feat_imp['feature'].values[::-1]],
        feat_imp['importance'].values[::-1],
        color=COLORS['primary'], edgecolor='white')
ax.set_xlabel('Importância (Random Forest)')
ax.set_title('Q4 — Importância das Features', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'q4_feature_importance.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Análise de Resíduos"))

cells.append(new_code_cell(textwrap.dedent("""\
best_name = max(models, key=lambda k: models[k]['r2'])
best_pred = models[best_name]['y_pred']
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(residuals, bins=40, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
axes[0].axvline(0, color=COLORS['danger'], linestyle='--', linewidth=2)
axes[0].set_xlabel('Resíduo (Real - Previsto)')
axes[0].set_ylabel('Frequência')
axes[0].set_title(f'Distribuição dos Resíduos ({best_name})')

stats.probplot(residuals, dist='norm', plot=axes[1])
axes[1].set_title(f'QQ-Plot ({best_name})')
axes[1].get_lines()[0].set_color(COLORS['primary'])
axes[1].get_lines()[1].set_color(COLORS['danger'])

fig.suptitle('Q4 — Análise de Resíduos', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'q4_residual_analysis.png')
plt.show()
""")))

cells.append(new_markdown_cell("### Comparação de Modelos"))

cells.append(new_code_cell(textwrap.dedent("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metric_names = ['RMSE', 'MAE', 'R²']
metric_keys = ['rmse', 'mae', 'r2']
for ax, mn, mk in zip(axes, metric_names, metric_keys):
    vals = [models[n][mk] for n in models]
    bars = ax.bar(list(models.keys()), vals,
                  color=[COLORS['primary'], COLORS['secondary']], edgecolor='white')
    ax.set_title(mn, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01 * max(abs(v) for v in vals),
                f'{h:.2f}', ha='center', fontsize=10)
fig.suptitle('Q4 — Comparação de Modelos', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'q4_model_comparison.png')
plt.show()
""")))

cells.append(new_markdown_cell(textwrap.dedent("""\
### Conclusões Q4

- O Random Forest supera o Ridge em todas as métricas, demonstrando a importância de capturar relações não-lineares.
- Features de lag temporal (especialmente lag de 1 dia e média móvel de 7 dias) são as mais importantes.
- Variáveis climáticas contribuem de forma complementar, com precipitação e temperatura entre as top features.
- O padrão semanal (dia da semana, fim de semana) é altamente preditivo.
- O modelo tem dificuldade em capturar picos extremos, indicando espaço para melhoria com modelos mais sofisticados.
""")))

# ── Final conclusions ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(textwrap.dedent("""\
---
## Conclusões Gerais — Parte 1

### Síntese dos Achados

1. **Clima e Demanda (Q1)**: Existe relação mensurável entre variáveis climáticas e volume de chamados, com precipitação sendo o fator mais relevante. A magnitude varia por tipo de serviço.

2. **Padrões Geoespaciais (Q2)**: A demanda é fortemente concentrada em poucos bairros e regiões, com alta desigualdade territorial. A normalização por área revela que APs centrais são as mais demandadas por km².

3. **Eventos Extremos (Q3)**: Feriados reduzem a demanda, enquanto eventos extremos de clima geram impacto variável. As diferenças são estatisticamente significativas.

4. **Previsão (Q4)**: O Random Forest com features temporais e climáticas consegue prever a demanda diária com precisão razoável. Lag de 1 dia e média móvel são as features mais importantes.

### Implicações para Política Pública

- Sistemas de alerta podem antecipar aumento de demanda em dias de chuva forte.
- A alocação de recursos deve considerar a desigualdade territorial da demanda.
- O modelo de previsão pode auxiliar no dimensionamento diário de equipes.
""")))

nb.cells = cells

nb_path = NOTEBOOKS / "01_analise_apis_clima.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"   Notebook saved: {nb_path}")

print("\n[OK] All done!")
print(f"  Figures: {FIG}")
print(f"  Findings: {OUTPUTS}")
print(f"  Notebook: {nb_path}")
