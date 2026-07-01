import os
import json
import gzip
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import re

from features import extract_features
from filters import is_honeypot, is_disqualified

# Base paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# Config
# Check for the new CSV dataset first, fallback to jsonl
csv_path = os.path.join(ROOT_DIR, "candidates_diverse_1000.csv")
json_path = os.path.join(ROOT_DIR, "candidates.jsonl")
CANDIDATES_FILE = csv_path if os.path.exists(csv_path) else json_path

MODEL_FILE = os.path.join(ROOT_DIR, "artifacts", "reranker_model.txt")
WEIGHTS_FILE = os.path.join(SRC_DIR, "weights.yaml")
OUTPUT_FILE = os.path.join(ROOT_DIR, "submission.csv")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_csv_row_to_json(row):
    cand = {'candidate_id': row['candidate_id']}
    cand['profile'] = {
        'summary': str(row.get('summary', '')),
        'years_of_experience': float(row.get('years_of_experience', 0)) if pd.notna(row.get('years_of_experience')) else 0.0,
        'current_title': str(row.get('current_title', '')),
        'location': str(row.get('location', ''))
    }
    
    career = []
    career_str = str(row.get('career_history', ''))
    if pd.notna(career_str) and career_str:
        for job in career_str.split(' || '):
            title = ""
            desc = ""
            duration = 0
            is_current = False
            company = ""
            
            if ' @ ' in job:
                parts = job.split(' @ ', 1)
                title = parts[0].strip()
                rest = parts[1]
                if ' (' in rest:
                    cparts = rest.split(' (', 1)
                    company = cparts[0].strip()
                    meta = '(' + cparts[1]
                else:
                    meta = rest
                
                if '): ' in meta:
                    dparts = meta.split('): ', 1)
                    desc = dparts[1].strip()
                    meta = dparts[0]
                
                dur_match = re.search(r'(\d+)mo', meta)
                if dur_match:
                    duration = int(dur_match.group(1))
                if 'Present' in meta:
                    is_current = True
            career.append({'title': title, 'company': company, 'description': desc, 'duration_months': duration, 'is_current': is_current})
    cand['career_history'] = career
    
    skills = []
    skills_str = str(row.get('skills', ''))
    if pd.notna(skills_str) and skills_str:
        for sk in skills_str.split(' || '):
            name = sk
            prof = 'beginner'
            dur = 0
            if ' (' in sk:
                name = sk.split(' (')[0].strip()
                if 'advanced' in sk: prof = 'advanced'
                elif 'expert' in sk: prof = 'expert'
                elif 'intermediate' in sk: prof = 'intermediate'
                
                dur_match = re.search(r'(\d+)mo', sk)
                if dur_match:
                    dur = int(dur_match.group(1))
            skills.append({'name': name, 'proficiency': prof, 'duration_months': dur})
    cand['skills'] = skills
    
    cand['redrob_signals'] = {
        'recruiter_response_rate': float(row.get('recruiter_response_rate', 0.0)) if pd.notna(row.get('recruiter_response_rate')) else 0.0,
        'interview_completion_rate': float(row.get('interview_completion_rate', 0.0)) if pd.notna(row.get('interview_completion_rate')) else 0.0,
        'willing_to_relocate': bool(row.get('willing_to_relocate', False)),
        'last_active_date': str(row.get('last_active_date', ''))
    }
    return cand

def yield_candidates(file_path):
    if file_path.endswith('.csv'):
        for chunk in pd.read_csv(file_path, chunksize=1000):
            for _, row in chunk.iterrows():
                yield parse_csv_row_to_json(row)
    else:
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.strip())

def load_fallback_weights():
    with open(WEIGHTS_FILE, 'r') as f:
        return yaml.safe_load(f)

def generate_reasoning(cand, score, rank):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    yoe = profile.get('years_of_experience', 0)
    title = profile.get('current_title', 'Engineer')
    
    advanced_skills = [s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']]
    skill_snippet = f"expertise in {advanced_skills[0]} and {advanced_skills[1]}" if len(advanced_skills) >= 2 else "relevant technical skills"
    
    rr_rate = signals.get('recruiter_response_rate', 0.0) * 100
    
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
        
    valid_candidates = []
    
    feature_names = [
        'semantic_similarity', 'skills_match', 'experience_relevance', 
        'career_trajectory', 'recency', 'engagement_signals', 'location_fit'
    ]
    
    logging.info(f"Processing candidates from {CANDIDATES_FILE}...")
    
    for cand in yield_candidates(CANDIDATES_FILE):
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
