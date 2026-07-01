import json, gzip, random, pandas as pd
import os

os.makedirs("../data", exist_ok=True)
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUT_FILE = "../data/labeled_candidates.csv"

total_lines = 100000
sample_indices = set(random.sample(range(total_lines), 200))

rows = []
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in sample_indices:
            cand = json.loads(line.strip())
            prof = cand.get('profile', {})
            skills = cand.get('skills', [])
            top_skills = ", ".join([s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:5])
            
            rows.append({
                'candidate_id': cand.get('candidate_id'),
                'label': '',  # USER FILLS THIS IN
                'current_title': prof.get('current_title', ''),
                'years_of_experience': prof.get('years_of_experience', 0),
                'top_skills': top_skills,
                'summary': prof.get('summary', '')[:200] + '...'
            })
            if len(rows) == 200:
                break

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)
print(f"Exported {len(df)} candidates to {OUT_FILE}")
