# Desafio Cientista de Dados Senior -- Multi-Agent Orchestration

## Project Overview
This is a Data Science Senior challenge for the Programa Pequenos Cariocas (PIC)
of Rio de Janeiro's city government. It uses 1746 public service call data
(14M+ records) to build exploratory analysis, predictive models, and a
prioritization system.

## Architecture
This project uses a multi-agent system coordinated through an Obsidian Vault
at `vault/`. Each agent has a dedicated CLAUDE.md defining its role.

## Agent Roster
| # | Agent | Vault Path | Scope |
|---|-------|------------|-------|
| 0 | Orchestrator | vault/00-orchestrator/ | Coordination, quality gates |
| 1 | Data Engineer | vault/01-data-engineer/ | BigQuery + API extraction |
| 2 | EDA Analyst | vault/02-eda-analyst/ | Part 1: Q1-Q4 |
| 3 | Feature Engineer | vault/03-feature-engineer/ | Q5: Feature engineering |
| 4 | Model Builder | vault/04-model-builder/ | Q6-Q8: Modeling |
| 5 | Prioritization Designer | vault/05-prioritization-designer/ | Q9-Q10 |
| 6 | Narrator | vault/06-narrator/ | Final docs, README, polish |

## How to Run an Agent
Each agent operates in its own Claude Code session. To invoke agent N:
1. Open a new Claude Code session
2. Set working directory to project root
3. Instruct: "You are Agent N. Read vault/0N-{name}/CLAUDE.md and execute your tasks."
4. The agent reads its CLAUDE.md, checks contracts/input-requirements.md, and proceeds.

## Parallel Execution
- Agents 2 and 3 can run simultaneously (Phase 2)
- Agent 6 can begin notebook 1 polish while Agent 5 is still working

## Conventions
- Language: Portuguese for all deliverables, English for code/technical docs
- Python: 3.10+, type hints encouraged
- Notebooks: clear markdown narrative between every code block
- Visualization: see vault/shared/conventions.md for palette and style

## Key Constraints
- BigQuery: ALWAYS filter with data_particao >= '2023-01-01'
- Sample size: 50K for modeling (Q5-Q10)
- Budget: top 20% prioritization (Q9-Q10)
- Train: 2023, Test: 2024 (temporal split, no random split)

## Data Pipeline
```
data/raw/        <-- Agent 1 writes (BigQuery + APIs)
data/processed/  <-- Agents 2,3 write (cleaned, merged)
data/features/   <-- Agent 3 writes (X_train, X_test, y_train, y_test)
results/models/  <-- Agent 4 writes (trained models, predictions)
results/figures/ <-- Agents 2,4,5 write (all visualizations)
```

## Quality Gates
See vault/00-orchestrator/quality-gates.md for pass/fail criteria between phases.

## .gitignore Policy
- Never commit files > 100MB
- data/raw/, data/processed/, data/features/ are gitignored
- results/models/*.joblib are gitignored
- Notebooks, src/, figures, and docs are committed
