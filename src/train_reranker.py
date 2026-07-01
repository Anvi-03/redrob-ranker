import os
import json
import gzip
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import yaml

from features import extract_features

LABELED_DATA_FILE = "../data/training_candidates_labeled.csv"
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
MODEL_OUTPUT_FILE = "../artifacts/reranker_model.txt"
REPORT_OUTPUT_FILE = "../artifacts/reranker_training_report.md"

def load_candidate_features_and_labels():
    if not os.path.exists(LABELED_DATA_FILE):
        raise FileNotFoundError(f"Labeled data not found at {LABELED_DATA_FILE}. Please run label_candidates.py first.")
        
    labels_df = pd.read_csv(LABELED_DATA_FILE)
    labeled_ids = set(labels_df['candidate_id'])
    
    candidates_data = {}
    open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open
    
    print(f"Scanning {CANDIDATES_FILE} for labeled candidates...")
    with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line.strip())
            cid = cand.get('candidate_id')
            if cid in labeled_ids:
                candidates_data[cid] = cand
                if len(candidates_data) == len(labeled_ids):
                    break

    # Build feature matrix
    X_rows = []
    y_rows = []
    
    # We define a fixed feature order based on our fallback weights list
    feature_names = [
        'semantic_similarity', 'skills_match', 'experience_relevance', 
        'career_trajectory', 'recency', 'engagement_signals', 'location_fit'
    ]
    
    for _, row in labels_df.iterrows():
        cid = row['candidate_id']
        label = row['label']
        if cid in candidates_data:
            feats = extract_features(candidates_data[cid])
            X_rows.append([feats.get(fn, 0.0) for fn in feature_names])
            y_rows.append(label)
            
    return np.array(X_rows), np.array(y_rows).round().astype(int), feature_names

def train_lambdarank():
    X, y, feature_names = load_candidate_features_and_labels()
    
    print(f"Loaded {len(X)} labeled examples.")
    if len(X) < 50:
        print("Warning: Very few labeled examples. Model may not generalize well.")
        
    os.makedirs("../artifacts", exist_ok=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_importances = []
    fold_ndcg10 = []
    fold_ndcg50 = []
    
    # Base LightGBM params for LambdaMART
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 50],
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_data_in_leaf': 5,
        'num_threads': 1,
        'verbose': -1
    }
    
    print("\nStarting 5-Fold Cross Validation...")
    
    best_model = None
    best_score = -1
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # In this hackathon there is only one "query" (the JD).
        # We simulate the query groups by assigning all items in the split to a single group.
        train_group = [len(X_train)]
        val_group = [len(X_val)]
        
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_group, feature_name=feature_names, reference=train_data)
        
        print(f"Training fold {fold+1}...")
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict on validation to manually calculate or extract metrics if needed
        # LightGBM records best_score
        val_metrics = model.best_score['valid_0']
        ndcg10 = val_metrics.get('ndcg@10', 0)
        ndcg50 = val_metrics.get('ndcg@50', 0)
        
        fold_ndcg10.append(ndcg10)
        fold_ndcg50.append(ndcg50)
        
        # Save feature importances
        importance = model.feature_importance(importance_type='gain')
        # Normalize importance
        if sum(importance) > 0:
            importance = importance / sum(importance)
        fold_importances.append(importance)
        
        print(f"Fold {fold+1}: NDCG@10 = {ndcg10:.4f}, NDCG@50 = {ndcg50:.4f}")
        
        # Keep track of the best overall model across folds to save (or we could retrain on all data)
        # We will retrain on ALL data after CV for the final model.

    print("\n--- CV Results ---")
    print(f"Mean NDCG@10: {np.mean(fold_ndcg10):.4f} (+/- {np.std(fold_ndcg10):.4f})")
    print(f"Mean NDCG@50: {np.mean(fold_ndcg50):.4f} (+/- {np.std(fold_ndcg50):.4f})")
    
    avg_importance = np.mean(fold_importances, axis=0)
    std_importance = np.std(fold_importances, axis=0)
    
    print("\nFeature Importances (Stability):")
    for fname, imp, std in zip(feature_names, avg_importance, std_importance):
        print(f"{fname}: {imp:.4f} (std: {std:.4f})")
        if std > 0.3:
            print(f"  WARNING: Feature {fname} has highly unstable importance. You may need more training data.")

    # Retrain on full dataset
    print("\nRetraining on full dataset for production...")
    full_group = [len(X)]
    full_data = lgb.Dataset(X, label=y, group=full_group, feature_name=feature_names)
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=100  # Fixed rounds since we can't early stop without a validation set
    )
    
    final_model.save_model(MODEL_OUTPUT_FILE)
    print(f"Model saved to {MODEL_OUTPUT_FILE}")
    
    # Generate Markdown Report
    with open(REPORT_OUTPUT_FILE, 'w') as f:
        f.write("# Reranker Training Report\n\n")
        f.write("This report documents the LambdaMART model trained for the Redrob Hackathon.\n\n")
        f.write("## Cross-Validation Performance (5-Fold)\n")
        f.write(f"- **Mean NDCG@10:** {np.mean(fold_ndcg10):.4f} ± {np.std(fold_ndcg10):.4f}\n")
        f.write(f"- **Mean NDCG@50:** {np.mean(fold_ndcg50):.4f} ± {np.std(fold_ndcg50):.4f}\n\n")
        
        f.write("## Feature Importances\n")
        f.write("| Feature | Mean Importance (Gain) | Std Dev |\n")
        f.write("|---------|------------------------|---------|\n")
        for fname, imp, std in sorted(zip(feature_names, avg_importance, std_importance), key=lambda x: x[1], reverse=True):
            f.write(f"| {fname} | {imp:.4f} | {std:.4f} |\n")
            
        f.write("\n\n*Note: High standard deviation indicates instability across folds, likely due to small dataset size.*\n")
    
    print(f"Report saved to {REPORT_OUTPUT_FILE}")

print("Script loaded")
if __name__ == "__main__":
    train_lambdarank()
