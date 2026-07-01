import pandas as pd
import json, gzip

# Read sample_submission.csv
sample_csv = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_submission.csv"
df_sample = pd.read_csv(sample_csv)

# Create a mapping of candidate_id to score
id_to_score = {}
for _, row in df_sample.iterrows():
    cid = row['candidate_id']
    score = float(row['score'])
    # format to 3 decimal places
    id_to_score[cid] = f"{score:.3f}"

# Extract these candidates from candidates.jsonl
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
rows = []
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        cand = json.loads(line.strip())
        cid = cand.get('candidate_id')
        if cid in id_to_score:
            prof = cand.get('profile', {})
            skills = cand.get('skills', [])
            career = cand.get('career_history', [])
            
            top_skills = ", ".join([s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:10])
            career_str = " | ".join([f"{j.get('title')} @ {j.get('company')} ({j.get('duration_months')}m)" for j in career[:3]])
            
            rows.append({
                'candidate_id': cid,
                'label': id_to_score[cid],
                'current_title': prof.get('current_title', ''),
                'years_of_experience': prof.get('years_of_experience', 0),
                'top_skills': top_skills,
                'career_history': career_str,
                'summary': prof.get('summary', '')
            })
            if len(rows) == len(id_to_score):
                break

# Save to data/labeled_candidates.csv
df_out = pd.DataFrame(rows)
df_out.to_csv("../data/labeled_candidates.csv", index=False)
print("Updated labels to use 3 decimal point scores from sample_submission.csv")
