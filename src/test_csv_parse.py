import pandas as pd
import re

def parse_csv_row_to_json(row):
    cand = {'candidate_id': row['candidate_id']}
    
    cand['profile'] = {
        'summary': str(row.get('summary', '')),
        'years_of_experience': float(row.get('years_of_experience', 0)) if pd.notna(row.get('years_of_experience')) else 0.0,
        'current_title': str(row.get('current_title', '')),
        'location': str(row.get('location', ''))
    }
    
    # Parse career history
    # Example: "ML Engineer @ Unacademy (2024-03-24 to Present, 26mo, HealthTech AI, 11-50 employees): Owned AWS... || ML Engineer @ Infosys..."
    career = []
    career_str = str(row.get('career_history', ''))
    if pd.notna(career_str) and career_str:
        jobs = career_str.split(' || ')
        for job in jobs:
            title = ""
            desc = ""
            duration = 0
            is_current = False
            
            # Extract title and rest
            if ' @ ' in job:
                parts = job.split(' @ ', 1)
                title = parts[0].strip()
                rest = parts[1]
                
                # Check description
                if '): ' in rest:
                    dparts = rest.split('): ', 1)
                    desc = dparts[1].strip()
                    meta = dparts[0]
                else:
                    meta = rest
                    
                # Extract duration
                dur_match = re.search(r'(\d+)mo', meta)
                if dur_match:
                    duration = int(dur_match.group(1))
                if 'Present' in meta:
                    is_current = True
                    
            career.append({
                'title': title,
                'description': desc,
                'duration_months': duration,
                'is_current': is_current
            })
    cand['career_history'] = career
    
    # Parse skills
    # Example: "LoRA (advanced, 7 endorsements, 60mo) || GANs (intermediate, 5 endorsements, 36mo)"
    skills = []
    skills_str = str(row.get('skills', ''))
    if pd.notna(skills_str) and skills_str:
        skill_parts = skills_str.split(' || ')
        for sk in skill_parts:
            name = sk
            prof = 'beginner'
            if ' (' in sk:
                name = sk.split(' (')[0].strip()
                if 'advanced' in sk: prof = 'advanced'
                elif 'expert' in sk: prof = 'expert'
                elif 'intermediate' in sk: prof = 'intermediate'
            skills.append({
                'name': name,
                'proficiency': prof
            })
    cand['skills'] = skills
    
    # Parse signals
    cand['redrob_signals'] = {
        'recruiter_response_rate': float(row.get('recruiter_response_rate', 0.0)) if pd.notna(row.get('recruiter_response_rate')) else 0.0,
        'interview_completion_rate': float(row.get('interview_completion_rate', 0.0)) if pd.notna(row.get('interview_completion_rate')) else 0.0,
        'willing_to_relocate': bool(row.get('willing_to_relocate', False)),
        'last_active_date': str(row.get('last_active_date', ''))
    }
    
    return cand

if __name__ == "__main__":
    df = pd.read_csv('candidates_diverse_1000.csv', nrows=2)
    for _, row in df.iterrows():
        print(parse_csv_row_to_json(row))
