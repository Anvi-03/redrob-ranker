import pandas as pd
import json

sub = pd.read_csv('submission.csv')
cands = set(sub['candidate_id'].head(10))

with open('candidates.jsonl', 'r') as f:
    for line in f:
        cand = json.loads(line)
        cid = cand.get('candidate_id')
        if cid in cands:
            print(f"--- {cid} ---")
            print(json.dumps(cand['career_history'], indent=2))
