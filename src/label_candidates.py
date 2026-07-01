import json
import gzip
import os
import random
import pandas as pd

# Constants
CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
LABELED_DATA_FILE = "../data/labeled_candidates.csv"
SAMPLE_SIZE = 250

def load_random_candidates(filepath, sample_size):
    print("Scanning candidates file to pick a random sample...")
    total_lines = 100000
    sample_indices = set(random.sample(range(total_lines), sample_size))
    
    sampled = []
    open_func = gzip.open if filepath.endswith('.gz') else open
    
    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in sample_indices:
                sampled.append(json.loads(line.strip()))
                if len(sampled) == sample_size:
                    break
    return sampled

def format_candidate_for_labeling(cand):
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    text = f"\n{'='*60}\n"
    text += f"ID: {cand.get('candidate_id')}\n"
    text += f"Title: {profile.get('current_title', 'N/A')} @ {profile.get('current_company', 'N/A')}\n"
    text += f"Exp: {profile.get('years_of_experience', 0)} years\n"
    text += f"Location: {profile.get('location', 'N/A')}\n"
    text += f"Summary: {profile.get('summary', 'N/A')[:300]}...\n\n"
    
    text += "--- CAREER HISTORY (Top 3) ---\n"
    for job in career[:3]:
        text += f"- {job.get('title')} at {job.get('company')} ({job.get('duration_months', 0)} months)\n"
    
    text += "\n--- TOP SKILLS ---\n"
    top_skills = [s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:10]
    text += f"{', '.join(top_skills)}\n"
    
    text += "\n--- BEHAVIORAL SIGNALS ---\n"
    text += f"Active 30d: {signals.get('last_active_date')}, Recruiter Reply Rate: {signals.get('recruiter_response_rate', 0)*100}%\n"
    
    return text

def main():
    os.makedirs('../data', exist_ok=True)
        
    print(f"--- Reranker Labeling Workflow ---")
    print("Label guide:")
    print("  3 = Perfect Fit (Strong ML, Production experience, Good behavioral signals)")
    print("  2 = Strong Fit (Good ML experience, mostly matches JD)")
    print("  1 = Borderline (Missing key elements but has potential)")
    print("  0 = Not a Fit (Pure research, purely consulting, lack of ML, or honeypot)\n")
    
    existing_labels = {}
    if os.path.exists(LABELED_DATA_FILE):
        df = pd.read_csv(LABELED_DATA_FILE)
        existing_labels = dict(zip(df['candidate_id'], df['label']))
        print(f"Found {len(existing_labels)} existing labels.")
    
    candidates = load_random_candidates(CANDIDATES_FILE, SAMPLE_SIZE)
    new_labels = []
    
    try:
        for cand in candidates:
            cid = cand['candidate_id']
            if cid in existing_labels:
                continue
                
            print(format_candidate_for_labeling(cand))
            
            while True:
                val = input("Enter label (0-3) or 'q' to quit: ").strip().lower()
                if val == 'q':
                    raise KeyboardInterrupt
                if val in ['0', '1', '2', '3']:
                    new_labels.append({'candidate_id': cid, 'label': int(val)})
                    break
                print("Invalid input. Please enter 0, 1, 2, 3 or 'q'.")
                
    except KeyboardInterrupt:
        print("\nSaving progress...")
        
    finally:
        if new_labels:
            df_new = pd.DataFrame(new_labels)
            if os.path.exists(LABELED_DATA_FILE):
                df_old = pd.read_csv(LABELED_DATA_FILE)
                df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['candidate_id'], keep='last')
            else:
                df_combined = df_new
                
            df_combined.to_csv(LABELED_DATA_FILE, index=False)
            print(f"Saved {len(new_labels)} new labels. Total labeled dataset size: {len(df_combined)}")
        else:
            print("No new labels to save.")

if __name__ == "__main__":
    main()
