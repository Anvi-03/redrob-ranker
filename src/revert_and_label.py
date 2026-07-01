import pandas as pd
import json, gzip, os

# 1. Delete the trained model to "revert back"
model_file = "../artifacts/reranker_model.txt"
if os.path.exists(model_file):
    os.remove(model_file)

# 2. Read sample_submission.csv
sample_csv = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_submission.csv"
df_sample = pd.read_csv(sample_csv)

# 3. Create a mapping of candidate_id to label based on rank
id_to_label = {}
for _, row in df_sample.iterrows():
    cid = row['candidate_id']
    rank = int(row['rank'])
    if rank <= 25:
        id_to_label[cid] = 3
    elif rank <= 50:
        id_to_label[cid] = 2
    elif rank <= 75:
        id_to_label[cid] = 1
    else:
        id_to_label[cid] = 0

# 4. Extract these 100 candidates from candidates.jsonl to make it easy to review
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
rows = []
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        cand = json.loads(line.strip())
        cid = cand.get('candidate_id')
        if cid in id_to_label:
            prof = cand.get('profile', {})
            skills = cand.get('skills', [])
            career = cand.get('career_history', [])
            
            top_skills = ", ".join([s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:10])
            career_str = " | ".join([f"{j.get('title')} @ {j.get('company')} ({j.get('duration_months')}m)" for j in career[:3]])
            
            rows.append({
                'candidate_id': cid,
                'label': id_to_label[cid],
                'rank_in_sample': id_to_label[cid], # Just for reference
                'current_title': prof.get('current_title', ''),
                'years_of_experience': prof.get('years_of_experience', 0),
                'top_skills': top_skills,
                'career_history': career_str,
                'summary': prof.get('summary', '')
            })
            if len(rows) == len(id_to_label):
                break

# 5. Save to data/labeled_candidates.csv
df_out = pd.DataFrame(rows)
df_out.to_csv("../data/labeled_candidates.csv", index=False)
print("Successfully reverted model and generated labels from sample_submission.csv")
