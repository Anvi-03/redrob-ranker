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
            # check for honeypot rules
            skills = cand.get('skills', [])
            expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0)
            
            yoe = cand.get('profile', {}).get('years_of_experience', 0)
            career = cand.get('career_history', [])
            max_dur = max([j.get('duration_months', 0) for j in career]) if career else 0
            
            reasons = []
            if expert_zero >= 3: reasons.append(f'expert_zero({expert_zero})')
            if max_dur > (yoe * 12) + 12: reasons.append(f'max_dur({max_dur} > yoe_months)')
            if not career: reasons.append('no_career')
            
            if reasons:
                honeypots.append((cid, reasons))

print(f"Found {len(honeypots)} honeypots in top 100: {honeypots}")
