import pandas as pd
import json

sub = pd.read_csv('submission.csv')
cands = set(sub['candidate_id'])

honeypots = []
with open('candidates.jsonl', 'r') as f:
    for line in f:
        cand = json.loads(line)
        cid = cand.get('candidate_id')
        if cid in cands:
            skills = cand.get('skills', [])
            expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
            if expert_zero >= 3:
                honeypots.append(cid)

print(f"Found {len(honeypots)} honeypots according to filters.py logic: {honeypots}")
