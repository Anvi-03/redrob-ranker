import re
import math
from datetime import datetime

# Key terms expected from the JD
ML_TERMS = {'machine learning', 'ml', 'ai', 'artificial intelligence', 'embedding', 'retrieval', 'ranking', 'llm', 'fine-tuning'}
PROD_TERMS = {'production', 'deploy', 'scale', 'infrastructure', 'pipeline'}
TECH_TERMS = {'python', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss', 'sentence-transformers', 'pytorch', 'tensorflow'}

def extract_features(cand):
    """
    Extracts raw features from a candidate profile, normalized to 0.0 - 1.0.
    Returns a dictionary of feature names to values.
    """
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})

    features = {}

    # 1. semantic_similarity (Proxy: Keyword density of ML/Prod/Tech terms in summary & career)
    text_blob = profile.get('summary', '').lower()
    for job in career:
        text_blob += " " + job.get('title', '').lower()
        text_blob += " " + job.get('description', '').lower()
    
    ml_matches = sum(1 for term in ML_TERMS if term in text_blob)
    prod_matches = sum(1 for term in PROD_TERMS if term in text_blob)
    tech_matches = sum(1 for term in TECH_TERMS if term in text_blob)
    
    # Normalize (arbitrary max bounds for normalization)
    f_ml = min(ml_matches / 5.0, 1.0)
    f_prod = min(prod_matches / 3.0, 1.0)
    f_tech = min(tech_matches / 5.0, 1.0)
    features['semantic_similarity'] = (f_ml + f_prod + f_tech) / 3.0

    # 2. skills_match
    jd_skills = ML_TERMS | PROD_TERMS | TECH_TERMS
    cand_skills = [s.get('name', '').lower() for s in skills]
    skill_hits = sum(1 for s in cand_skills if any(j_skill in s for j_skill in jd_skills))
    features['skills_match'] = min(skill_hits / 8.0, 1.0)

    # 3. experience_relevance
    yoe = profile.get('years_of_experience', 0)
    # JD says 5-9 years is the sweet spot. 
    # Let's map 5-9 to 1.0, 3-4 to 0.7, 10+ to 0.8, else 0.4
    if 5 <= yoe <= 9:
        features['experience_relevance'] = 1.0
    elif 3 <= yoe < 5:
        features['experience_relevance'] = 0.7
    elif yoe > 9:
        features['experience_relevance'] = 0.8
    else:
        features['experience_relevance'] = max(yoe / 10.0, 0.1) # Smooth falloff

    # 4. career_trajectory (Penalty for "title chasers" jumping every 1.5 yrs)
    # Average duration of past jobs. If < 18 months, penalize.
    durations = [job.get('duration_months', 0) for job in career if not job.get('is_current', False)]
    if durations:
        avg_tenure = sum(durations) / len(durations)
        features['career_trajectory'] = min(max((avg_tenure - 12) / 24.0, 0.0), 1.0) # 1.0 if > 36 months avg, 0.0 if < 12
    else:
        features['career_trajectory'] = 0.5 # Neutral if no history

    # 5. engagement_signals
    rr_rate = signals.get('recruiter_response_rate', 0.0)
    ic_rate = signals.get('interview_completion_rate', 0.0)
    # Combine response rate and interview completion
    features['engagement_signals'] = (rr_rate + ic_rate) / 2.0

    # 6. location_fit
    loc = profile.get('location', '').lower()
    will_relocate = signals.get('willing_to_relocate', False)
    if 'pune' in loc or 'noida' in loc or 'delhi' in loc or 'ncr' in loc or 'mumbai' in loc:
        features['location_fit'] = 1.0
    elif will_relocate:
        features['location_fit'] = 0.8
    else:
        features['location_fit'] = 0.0

    # 7. Recency of activity
    # Assuming current year is 2026 based on hackathon metadata or system time. Let's just use days since last active if possible.
    # If date parsing fails, fallback.
    last_active = signals.get('last_active_date', '2026-06-30')
    try:
        dt = datetime.strptime(last_active, '%Y-%m-%d')
        # Days from 2026-06-30
        delta = (datetime(2026, 6, 30) - dt).days
        features['recency'] = max(1.0 - (delta / 180.0), 0.0) # 1.0 if today, 0.0 if 6+ months ago
    except:
        features['recency'] = 0.5

    return features
