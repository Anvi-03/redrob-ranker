import pandas as pd
import json
import sys
sys.path.append('src')
from filters import is_disqualified

sub = pd.read_csv('submission.csv')
cands = set(sub['candidate_id'])

disq = []
with open('candidates.jsonl', 'r') as f:
    for line in f:
        cand = json.loads(line)
        cid = cand.get('candidate_id')
        if cid in cands:
            if is_disqualified(cand):
                disq.append(cid)

print(f"Found {len(disq)} disqualified according to filters.py logic: {disq}")
