# Project Conventions

## Language
- **Deliverables** (notebooks, README, figures): Portuguese (pt-BR)
- **Code** (variables, functions, comments): English
- **Technical docs** (CLAUDE.md, contracts): English

## Python Style
- Python 3.10+
- Type hints encouraged but not required
- f-strings preferred over .format()
- Meaningful variable names (no single letters except in loops)

## Visualization Palette

### Primary Colors (Rio de Janeiro / PIC theme)
```python
COLORS = {
    'primary': '#1B4F72',      # Dark blue (trust, government)
    'secondary': '#2E86C1',    # Medium blue
    'accent': '#F39C12',       # Gold/amber (attention, priority)
    'success': '#27AE60',      # Green (resolved)
    'danger': '#E74C3C',       # Red (delayed, at-risk)
    'neutral': '#95A5A6',      # Gray
    'light': '#D5E8D4',        # Light green
}

# Sequential palette for heatmaps
SEQUENTIAL = 'YlOrRd'  # Yellow-Orange-Red

# Diverging palette for correlations
DIVERGING = 'RdBu_r'   # Red-Blue reversed

# Categorical palette
CATEGORICAL = ['#1B4F72', '#2E86C1', '#F39C12', '#27AE60', '#E74C3C',
               '#8E44AD', '#E67E22', '#16A085', '#2C3E50', '#D35400']
```

### Matplotlib Defaults
```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
```

## Notebook Structure
Each notebook follows this structure:
1. **Resumo Executivo** (Executive Summary) -- markdown cell
2. **Imports e Configuracao** -- single code cell with all imports
3. **Carregamento dos Dados** -- load data section
4. **Analise / Modelagem** -- main content sections
5. **Conclusoes** -- markdown summary cell

## File Naming
- Notebooks: `01_descricao_curta.ipynb` (numbered, snake_case)
- Figures: `q{N}_{descricao}.png` or `.html` for interactive
- Data: `{descricao}_{periodo}.{ext}` (e.g., `chamados_2023_2024.parquet`)
- Models: `{model_name}.joblib`

## Git Conventions
- Never commit files > 100MB
- data/raw/, data/processed/, data/features/ are gitignored
- Commit messages in English, prefixed: feat:, fix:, docs:, refactor:
