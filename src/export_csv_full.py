import json, gzip, random, pandas as pd
import os

CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUT_FILE = "../data/labeled_candidates.csv"

# Load existing so we don't lose any labels the user might have typed
existing_labels = {}
if os.path.exists(OUT_FILE):
    df_existing = pd.read_csv(OUT_FILE)
    for _, row in df_existing.iterrows():
        if pd.notna(row['label']) and str(row['label']).strip() != "":
            existing_labels[row['candidate_id']] = row['label']
            
    # Get the same candidate IDs to overwrite
    sample_ids = set(df_existing['candidate_id'].tolist())
else:
    sample_ids = set()

rows = []
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        cand = json.loads(line.strip())
        if cand.get('candidate_id') in sample_ids:
            prof = cand.get('profile', {})
            skills = cand.get('skills', [])
            career = cand.get('career_history', [])
            
            top_skills = ", ".join([s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:10])
            
            career_str = " | ".join([f"{j.get('title')} @ {j.get('company')} ({j.get('duration_months')}m)" for j in career[:3]])
            
            rows.append({
                'candidate_id': cand.get('candidate_id'),
                'label': existing_labels.get(cand.get('candidate_id'), ''),
                'current_title': prof.get('current_title', ''),
                'years_of_experience': prof.get('years_of_experience', 0),
                'top_skills': top_skills,
                'career_history': career_str,
                'summary': prof.get('summary', '')  # FULL SUMMARY, NO TRUNCATION
            })
            if len(rows) == len(sample_ids):
                break

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)
print(f"Exported {len(df)} candidates to {OUT_FILE} with full text.")
