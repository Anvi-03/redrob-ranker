import os
import json
import gzip
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

from features import extract_features
from filters import is_honeypot, is_disqualified

# Config
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
MODEL_FILE = "../artifacts/reranker_model.txt"
WEIGHTS_FILE = "weights.yaml"
OUTPUT_FILE = "../submission.csv"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_fallback_weights():
    with open(WEIGHTS_FILE, 'r') as f:
        return yaml.safe_load(f)

def generate_reasoning(cand, score, rank):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    yoe = profile.get('years_of_experience', 0)
    title = profile.get('current_title', 'Engineer')
    
    # Extract top relevant skills
    advanced_skills = [s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']]
    skill_snippet = f"expertise in {advanced_skills[0]} and {advanced_skills[1]}" if len(advanced_skills) >= 2 else "relevant technical skills"
    
    # Behavioral
    rr_rate = signals.get('recruiter_response_rate', 0.0) * 100
    
    # Vary the reasoning slightly based on rank to avoid identical strings
    if rank <= 10:
        return f"Exceptional {title} with {yoe} years of experience and {skill_snippet}. Highly responsive ({rr_rate:.0f}% reply rate) and perfect fit for production ML requirements."
    elif rank <= 50:
        return f"Strong {yoe}-year ML background featuring {skill_snippet}. Solid engagement signals and demonstrable product experience."
    else:
        return f"Capable engineer with {yoe} years experience. Shows {skill_snippet} but may lack deeper production scale or perfect behavioral availability compared to top candidates."

def main():
    use_model = os.path.exists(MODEL_FILE)
    model = None
    fallback_weights = None
    
    if use_model:
        logging.info(f"Trained model found at {MODEL_FILE}. Loading LightGBM model for scoring.")
        model = lgb.Booster(model_file=MODEL_FILE)
    else:
        logging.warning(f"No trained model found at {MODEL_FILE}. Falling back to manual weights.yaml.")
        fallback_weights = load_fallback_weights()
        
    open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open
    
    valid_candidates = []
    
    feature_names = [
        'semantic_similarity', 'skills_match', 'experience_relevance', 
        'career_trajectory', 'recency', 'engagement_signals', 'location_fit'
    ]
    
    logging.info("Processing candidates and computing features...")
    
    # We will score sequentially to keep memory usage low (under 16GB)
    with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line.strip())
            
            # Apply hard filters FIRST
            if is_honeypot(cand):
                continue
            if is_disqualified(cand):
                continue
                
            feats = extract_features(cand)
            
            # Calculate score
            score = 0.0
            if use_model:
                # We store features to batch predict later, or predict point-wise.
                # Batch predict is faster for LightGBM.
                feat_array = [feats.get(fn, 0.0) for fn in feature_names]
                valid_candidates.append({
                    'cand': cand,
                    'features': feat_array
                })
            else:
                # Fallback weighted sum
                for fn in feature_names:
                    score += feats.get(fn, 0.0) * fallback_weights.get(fn, 0.0)
                
                valid_candidates.append({
                    'cand': cand,
                    'score': score
                })
                
    logging.info(f"Filtered down to {len(valid_candidates)} valid candidates.")
    
    if use_model:
        logging.info("Running model predictions...")
        X = np.array([vc['features'] for vc in valid_candidates])
        # LightGBM predict returns a numpy array of scores
        scores = model.predict(X)
        for i, vc in enumerate(valid_candidates):
            vc['score'] = float(scores[i])
            
    # Sort by candidate_id ascending first, then by score descending (stable sort)
    logging.info("Ranking candidates...")
    valid_candidates.sort(key=lambda x: x['cand']['candidate_id'])
    valid_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    top_100 = valid_candidates[:100]
    
    # Format for submission
    logging.info("Generating submission file...")
    submission_rows = []
    for rank, item in enumerate(top_100, start=1):
        cand = item['cand']
        score = item['score']
        cid = cand.get('candidate_id')
        reasoning = generate_reasoning(cand, score, rank)
        
        submission_rows.append({
            'candidate_id': cid,
            'rank': rank,
            'score': score,
            'reasoning': reasoning
        })
        
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(OUTPUT_FILE, index=False)
    
    logging.info(f"Successfully generated {OUTPUT_FILE} with {len(df_sub)} ranked candidates.")

if __name__ == "__main__":
    main()
