# Task: Q9 Priority Score Design

## Steps
1. Load test predictions and feature data
2. Define urgency_score based on chamado type categorization
3. Define equity_score based on bairro historical resolution rates
4. Define context_score based on weather conditions
5. Compute P(delay) = 1 - y_proba
6. Combine into priority_score with calibrated weights
7. Normalize to [0, 1] range
8. Analyze score distribution and component contributions
9. Document formula and weight justification
