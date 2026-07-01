# Reranker Training Report

This report documents the LambdaMART model trained for the Redrob Hackathon.

## Cross-Validation Performance (5-Fold)
- **Mean NDCG@10:** 0.9283 ± 0.0479
- **Mean NDCG@50:** 0.8955 ± 0.0210

## Feature Importances
| Feature | Mean Importance (Gain) | Std Dev |
|---------|------------------------|---------|
| semantic_similarity | 0.8623 | 0.0448 |
| experience_relevance | 0.0661 | 0.0352 |
| skills_match | 0.0271 | 0.0169 |
| recency | 0.0220 | 0.0068 |
| career_trajectory | 0.0123 | 0.0040 |
| engagement_signals | 0.0062 | 0.0036 |
| location_fit | 0.0041 | 0.0054 |


*Note: High standard deviation indicates instability across folds, likely due to small dataset size.*
