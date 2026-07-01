import json
import gzip
from collections import Counter

CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

skill_counts = Counter()
summary_lengths = Counter()

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        cand = json.loads(line.strip())
        skills = cand.get('skills', [])
        summary = cand.get('profile', {}).get('summary', '')
        
        skill_counts[len(skills)] += 1
        # count words in summary
        summary_lengths[len(summary.split()) // 10 * 10] += 1
        
print("Skill count distribution:")
for k, v in sorted(skill_counts.items(), key=lambda x: x[0]):
    print(f"{k} skills: {v} candidates")
    
print("\nSummary length (words) distribution:")
for k, v in sorted(summary_lengths.items(), key=lambda x: x[0]):
    print(f"{k}-{k+9} words: {v} candidates")
