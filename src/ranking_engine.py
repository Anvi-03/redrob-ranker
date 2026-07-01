"""
ranking_engine.py — Production-grade candidate ranking engine for Redrob Challenge.

Multi-stage architecture:
  Stage 1: Hard filters (honeypots, disqualifiers)
  Stage 2: Rich feature extraction (~25 features)
  Stage 3: Weighted composite scoring
  Stage 4: Dynamic reasoning generation

Designed to work standalone (heuristic scoring) or with pre-computed
sentence-transformer embeddings + FAISS for semantic retrieval.
"""

import re
import math
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# JD-DERIVED TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

# High-priority retrieval and ranking specific skills
RETRIEVAL_RANKING_SKILLS = {
    'retrieval', 'rag', 'retrieval-augmented generation', 'semantic search',
    'hybrid search', 'vector search', 'bm25', 'dense retrieval',
    'ranking', 'learning to rank', 're-ranking', 'reranking',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'elasticsearch', 'elastic search', 'faiss', 'chroma', 'chromadb', 'pgvector',
    'embeddings', 'embedding', 'sentence-transformers', 'sentence transformers',
    'openai embeddings', 'bge', 'e5',
}

EVALUATION_METRICS = {
    'ndcg', 'mrr', 'map', 'a/b testing', 'evaluation framework', 'offline evaluation'
}

# The JD says you MUST have these (now excluding the specific retrieval ones above)
MUST_HAVE_SKILLS = {
    'llm', 'llms', 'large language models', 'large language model',
} | RETRIEVAL_RANKING_SKILLS | EVALUATION_METRICS

# Removed generic frameworks from nice-to-have, kept specific finetuning/serving
NICE_TO_HAVE_SKILLS = {
    'fine-tuning', 'fine tuning', 'finetuning', 'lora', 'qlora', 'peft',
    'xgboost', 'lightgbm', 'gradient boosting',
    'recommendation', 'recommendation systems', 'recommendation engine',
    'distributed systems', 'inference optimization', 'model serving',
    'hr-tech', 'hrtech', 'recruiting', 'talent',
}

# Langchain wrapper specific (used for penalty if no foundational skills)
LANGCHAIN_WRAPPER_SKILLS = {
    'langchain', 'llamaindex', 'llama index', 'haystack'
}

# Generic AI / Frameworks (Demoted in weight)
CORE_ML = {
    'machine learning', 'deep learning', 'nlp',
    'natural language processing', 'transformers', 'transformer',
    'neural networks', 'neural network', 'pytorch', 'tensorflow',
    'information retrieval', 'data science', 'scikit-learn', 'sklearn',
    'hugging face', 'huggingface',
}

PRODUCTION_SKILLS = {
    'python', 'docker', 'kubernetes', 'k8s', 'mlops', 'mlflow',
    'fastapi', 'flask', 'django', 'aws', 'gcp', 'azure',
    'ci/cd', 'cicd', 'deployment', 'production', 'microservices',
    'kafka', 'redis', 'postgresql', 'mongodb', 'airflow',
}

WRONG_DOMAIN = {
    'computer vision', 'image classification', 'object detection',
    'speech recognition', 'tts', 'text-to-speech', 'asr',
    'robotics', 'autonomous driving', 'self-driving',
    'gans', 'image segmentation', 'yolo', 'opencv',
    'diffusion models', 'stable diffusion', 'image generation',
}

# ── Title taxonomy ──
PERFECT_TITLES = {
    'ai engineer', 'senior ai engineer', 'staff ai engineer',
    'ml engineer', 'senior ml engineer', 'staff ml engineer',
    'machine learning engineer', 'senior machine learning engineer',
    'applied scientist', 'applied ml scientist',
    'nlp engineer', 'senior nlp engineer',
    'search engineer', 'senior search engineer',
    'ranking engineer', 'recommendation engineer',
    'ai/ml engineer', 'ml/ai engineer',
}

GOOD_TITLES = {
    'data scientist', 'senior data scientist', 'staff data scientist',
    'research engineer', 'senior research engineer',
    'ai research engineer', 'ai specialist', 'ml specialist',
    'junior ml engineer', 'junior ai engineer',
    'deep learning engineer', 'senior deep learning engineer',
}

ADJACENT_TITLES = {
    'software engineer', 'senior software engineer',
    'backend engineer', 'senior backend engineer',
    'full stack developer', 'senior full stack developer',
    'data engineer', 'senior data engineer',
    'platform engineer', 'senior platform engineer',
    'tech lead', 'engineering manager',
    'solutions architect', 'cloud engineer',
}

IRRELEVANT_TITLES = {
    'marketing manager', 'hr manager', 'human resources',
    'sales executive', 'sales manager', 'business development',
    'content writer', 'copywriter', 'content strategist',
    'graphic designer', 'ui designer', 'ux designer',
    'accountant', 'financial analyst', 'finance manager',
    'customer support', 'customer success',
    'operations manager', 'supply chain',
    'civil engineer', 'mechanical engineer', 'electrical engineer',
    'chemical engineer', 'structural engineer',
    'project manager', 'program manager', 'scrum master',
    'business analyst', 'management consultant',
    'teacher', 'professor', 'lecturer',
    'lawyer', 'legal counsel', 'advocate',
    'doctor', 'physician', 'nurse',
    'chef', 'hospitality', 'hotel manager',
    'qa engineer', 'test engineer', 'quality analyst',
    'frontend engineer', 'react developer', 'angular developer',
    'devops engineer',
}

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree',
    'mphasis', 'l&t infotech', 'persistent systems',
    'deloitte', 'pwc', 'kpmg', 'ey', 'ernst & young',
}

PRODUCT_COMPANIES = {
    'razorpay', 'swiggy', 'zomato', 'flipkart', 'cred', 'ola', 'uber',
    'paytm', 'phonepe', 'dream11', 'meesho', 'byjus', 'unacademy',
    'freshworks', 'zoho', 'postman', 'haptik', 'rephrase.ai', 'sarvam',
    'google', 'microsoft', 'amazon', 'meta', 'apple', 'netflix',
    'airbnb', 'stripe', 'shopify', 'spotify', 'linkedin', 'twitter',
    'salesforce', 'adobe', 'nvidia', 'intel', 'ibm',
    'grab', 'gojek', 'tokopedia', 'shopee',
    'atlassian', 'canva', 'notion', 'figma', 'vercel',
    # Fictional companies from the dataset
    'stark industries', 'wayne enterprises', 'pied piper', 'hooli',
    'acme corp', 'initech', 'globex inc', 'dunder mifflin',
    'umbrella corp', 'cyberdyne', 'aperture science',
}

GOOD_LOCATIONS = {
    'pune', 'noida', 'delhi', 'ncr', 'new delhi',
    'mumbai', 'bombay', 'hyderabad', 'secunderabad',
    'bangalore', 'bengaluru', 'gurgaon', 'gurugram',
}

CAREER_PROD_KEYWORDS = [
    'production', 'deployed', 'shipped', 'launched', 'scale', 'scaled',
    'ranking', 'search', 'retrieval', 'embedding', 'recommendation',
    'inference', 'model serving', 'real users', 'pipeline',
    'vector', 'llm', 'fine-tun', 'a/b test', 'latency',
    'microservice', 'api', 'end-to-end', 'system design',
]

CS_FIELDS = {
    'computer science', 'cs', 'artificial intelligence', 'ai',
    'machine learning', 'ml', 'data science', 'statistics',
    'mathematics', 'math', 'computational', 'informatics',
    'information technology', 'it', 'software engineering',
    'electronics', 'electrical engineering', 'ece',
}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: HARD FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_honeypot(cand):
    """Detect honeypot profiles with logically impossible data."""
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    yoe = profile.get('years_of_experience', 0)

    # 1. Expert proficiency with 0 months used
    expert_zero = sum(1 for s in skills
                      if s.get('proficiency') == 'expert'
                      and s.get('duration_months', 1) == 0)
    if expert_zero >= 3:
        return True, f"expert_zero_months({expert_zero} skills)"

    # 2. Single job duration > total YoE
    yoe_months = yoe * 12
    for job in career:
        dur = job.get('duration_months', 0)
        if dur > yoe_months + 12:
            return True, f"impossible_tenure({dur}mo vs {yoe}yr)"

    # 3. Career/YoE ratio wildly off
    total_career = sum(j.get('duration_months', 0) for j in career)
    if total_career > 0 and yoe > 0:
        ratio = total_career / (yoe * 12)
        if ratio > 2.5 or ratio < 0.2:
            return True, f"career_yoe_mismatch(ratio={ratio:.2f})"

    # 4. Expert in 10+ skills with < 3 yrs
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count >= 10 and yoe < 3:
        return True, f"impossible_expertise({expert_count} expert, {yoe}yr)"

    # 5. Skills count impossibly high relative to career
    if len(skills) > 20 and len(career) <= 1 and yoe < 2:
        return True, f"skill_inflation({len(skills)} skills, {yoe}yr, {len(career)} jobs)"

    return False, ""


def is_hard_disqualified(cand):
    """Rule-based disqualifications from the JD."""
    career = cand.get('career_history', [])
    if not career:
        return True, "no_career_history"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: RICH FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _fuzzy_match_count(target_set, source_set):
    """Count how many items from target_set appear in any item of source_set."""
    hits = 0
    for t in target_set:
        for s in source_set:
            if t in s or s in t:
                hits += 1
                break
    return hits


def _text_match_count(target_set, text):
    """Count how many items from target_set appear in the text blob."""
    count = 0
    text_lower = text.lower()
    for t in target_set:
        if t in text_lower:
            count += 1
    return count


def extract_rich_features(cand):
    """
    Extract ~25 features from a candidate profile, each normalized 0-1.
    Returns a dict of feature_name -> value.
    """
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    signals = cand.get('redrob_signals', {})
    education = cand.get('education', [])

    yoe = profile.get('years_of_experience', 0)
    title = profile.get('current_title', '').lower().strip()
    summary = profile.get('summary', '').lower()
    location = profile.get('location', '').lower()

    # Build text blob from career
    career_text = ""
    for job in career:
        career_text += " " + job.get('title', '').lower()
        career_text += " " + job.get('description', '').lower()
    full_text = summary + " " + career_text

    skill_names = {s.get('name', '').lower() for s in skills}
    adv_expert_skills = {s.get('name', '').lower() for s in skills
                         if s.get('proficiency') in ('advanced', 'expert')}

    features = {}

    # ── 1. TITLE RELEVANCE (0-1.0) ──
    title_score = 0.0
    for t in PERFECT_TITLES:
        if t == title or title.startswith(t) or title.endswith(t) or t in title:
            title_score = 1.0
            break
    if title_score == 0:
        for t in GOOD_TITLES:
            if t == title or title.startswith(t) or title.endswith(t) or t in title:
                title_score = 0.7
                break
    if title_score == 0:
        for t in ADJACENT_TITLES:
            if t == title or title.startswith(t) or title.endswith(t) or t in title:
                title_score = 0.35
                break
    if title_score == 0:
        for t in IRRELEVANT_TITLES:
            if t == title or title.startswith(t) or t in title:
                title_score = 0.0
                break
        # If not found in any list, give a small score (unknown title)
        if title_score == 0 and title:
            # Check if title contains any ML-related words
            ml_words = {'ml', 'ai', 'machine learning', 'data', 'engineer', 'scientist'}
            if any(w in title for w in ml_words):
                title_score = 0.4
            else:
                title_score = 0.1
    features['title_relevance'] = title_score

    # ── NEW: RETRIEVAL & RANKING SKILLS (0-1.0) ──
    ret_skill_hits = _fuzzy_match_count(RETRIEVAL_RANKING_SKILLS, skill_names)
    ret_text_hits = _text_match_count(RETRIEVAL_RANKING_SKILLS, full_text)
    features['retrieval_ranking_skills'] = min((ret_skill_hits + ret_text_hits * 0.5) / 5.0, 1.0)

    # ── NEW: EVALUATION METRICS (0-1.0) ──
    eval_skill_hits = _fuzzy_match_count(EVALUATION_METRICS, skill_names)
    eval_text_hits = _text_match_count(EVALUATION_METRICS, full_text)
    features['evaluation_metrics'] = min((eval_skill_hits + eval_text_hits * 0.5) / 2.0, 1.0)

    # ── 2. MUST-HAVE SKILLS (0-1.0) ──
    must_skill_hits = _fuzzy_match_count(MUST_HAVE_SKILLS, skill_names)
    must_text_hits = _text_match_count(MUST_HAVE_SKILLS, full_text)
    # Weight skill list matches higher than text mentions
    must_combined = must_skill_hits * 1.0 + must_text_hits * 0.5
    features['must_have_skills'] = min(must_combined / 6.0, 1.0)

    # ── 3. MUST-HAVE with proficiency weighting ──
    prof_weighted = 0
    for s in skills:
        sname = s.get('name', '').lower()
        prof = s.get('proficiency', 'beginner')
        dur = s.get('duration_months', 0)
        is_must = any(t in sname or sname in t for t in MUST_HAVE_SKILLS)
        if is_must:
            weight = {'expert': 1.0, 'advanced': 0.8, 'intermediate': 0.5, 'beginner': 0.2}.get(prof, 0.2)
            # Penalize skills with 0 duration (claimed but never used)
            if dur == 0:
                weight *= 0.3
            prof_weighted += weight
    features['must_have_proficiency'] = min(prof_weighted / 4.0, 1.0)

    # ── 4. NICE-TO-HAVE SKILLS (0-1.0) ──
    nice_skill_hits = _fuzzy_match_count(NICE_TO_HAVE_SKILLS, skill_names)
    nice_text_hits = _text_match_count(NICE_TO_HAVE_SKILLS, full_text)
    features['nice_to_have_skills'] = min((nice_skill_hits + nice_text_hits * 0.5) / 4.0, 1.0)

    # ── 5. CORE ML FOUNDATION (0-1.0) ──
    core_hits = _fuzzy_match_count(CORE_ML, skill_names)
    core_text = _text_match_count(CORE_ML, full_text)
    features['core_ml'] = min((core_hits + core_text * 0.5) / 5.0, 1.0)

    # ── 6. PRODUCTION SKILLS (0-1.0) ──
    prod_hits = _fuzzy_match_count(PRODUCTION_SKILLS, skill_names)
    features['production_skills'] = min(prod_hits / 5.0, 1.0)

    # ── 7. EXPERIENCE FIT (0-1.0) ──
    if 5 <= yoe <= 9:
        features['experience_fit'] = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        features['experience_fit'] = 0.7
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        features['experience_fit'] = 0.4
    else:
        features['experience_fit'] = max(0.1, min(yoe / 15.0, 0.3))

    # ── 8. CAREER QUALITY: Product company experience ──
    has_product = False
    product_months = 0
    for job in career:
        comp = job.get('company', '').lower()
        if any(pc in comp for pc in PRODUCT_COMPANIES):
            has_product = True
            product_months += job.get('duration_months', 0)
    features['product_company'] = min(product_months / 60.0, 1.0) if has_product else 0.0

    # ── 9. CAREER QUALITY: Production ML evidence in descriptions ──
    prod_ml_evidence = 0
    for job in career:
        desc = job.get('description', '').lower()
        for kw in CAREER_PROD_KEYWORDS:
            if kw in desc:
                prod_ml_evidence += 1
    features['career_ml_evidence'] = min(prod_ml_evidence / 8.0, 1.0)

    # ── 10. CAREER QUALITY: ML titles across career (not just current) ──
    ml_title_count = 0
    for job in career:
        jtitle = job.get('title', '').lower()
        ml_words = {'ml', 'ai', 'machine learning', 'data scientist', 'nlp',
                    'search', 'ranking', 'recommendation', 'deep learning'}
        if any(w in jtitle for w in ml_words):
            ml_title_count += 1
    features['career_ml_titles'] = min(ml_title_count / 3.0, 1.0)

    # ── 11. CAREER STABILITY (anti title-chaser) ──
    non_current = [j.get('duration_months', 0) for j in career if not j.get('is_current', False)]
    if non_current and len(non_current) >= 2:
        avg_tenure = sum(non_current) / len(non_current)
        features['career_stability'] = min(max((avg_tenure - 12) / 24.0, 0.0), 1.0)
    else:
        features['career_stability'] = 0.5

    # ── 12. KEYWORD STUFFING DETECTION ──
    stuffing_score = 0.0

    # 12a. Ratio of advanced/expert skills to YoE
    adv_count = len(adv_expert_skills)
    if yoe > 0 and adv_count / yoe > 3.0:
        stuffing_score += 0.3

    # 12b. Skills listed but with 0 duration
    zero_dur = sum(1 for s in skills if s.get('duration_months', 1) == 0
                   and s.get('proficiency') in ('advanced', 'expert'))
    if zero_dur >= 3:
        stuffing_score += 0.3

    # 12c. Title is irrelevant but has many ML skills
    title_irrelevant = any(t in title for t in IRRELEVANT_TITLES) if title else False
    ml_skill_count = _fuzzy_match_count(MUST_HAVE_SKILLS | CORE_ML, skill_names)
    if title_irrelevant and ml_skill_count >= 5:
        stuffing_score += 0.4

    # 12d. Career descriptions show no ML work but skills list is full of ML
    if ml_skill_count >= 5 and prod_ml_evidence == 0 and ml_title_count == 0:
        stuffing_score += 0.3

    features['keyword_stuffing'] = min(stuffing_score, 1.0)  # Higher = more stuffed

    # ── 13. WRONG DOMAIN PENALTY ──
    wrong_hits = _fuzzy_match_count(WRONG_DOMAIN, skill_names)
    wrong_text = _text_match_count(WRONG_DOMAIN, full_text)
    wrong_score = min((wrong_hits + wrong_text * 0.3) / 4.0, 1.0)
    # Mitigate penalty if they actually have strong NLP/IR overlap
    if features.get('retrieval_ranking_skills', 0) > 0.5:
        wrong_score *= 0.3 # Reduce penalty drastically
    features['wrong_domain'] = wrong_score

    # ── NEW PENALTY: CONSULTING ONLY ──
    all_consulting = True
    for job in career:
        comp = job.get('company', '').lower()
        if not any(cf in comp for cf in CONSULTING_FIRMS):
            all_consulting = False
            break
    features['consulting_only'] = 1.0 if (all_consulting and len(career) >= 1) else 0.0

    # ── NEW PENALTY: PURE RESEARCH ──
    academic_terms = {'researcher', 'postdoc', 'phd', 'graduate research',
                      'research assistant', 'fellow', 'professor', 'lecturer'}
    all_academic = all(
        any(at in job.get('title', '').lower() for at in academic_terms)
        for job in career
    )
    features['pure_research'] = 1.0 if (all_academic and len(career) >= 1) else 0.0

    # ── NEW PENALTY: ARCHITECTURE ONLY ──
    # If the latest job is "architect" and there's no mention of python/coding/deployment recently
    features['architecture_only'] = 0.0
    if career:
        latest_job = career[0]
        ltitle = latest_job.get('title', '').lower()
        ldesc = latest_job.get('description', '').lower()
        if 'architect' in ltitle:
            if not any(ps in ldesc for ps in {'python', 'pytorch', 'tensorflow', 'code', 'deploy', 'pipeline', 'infrastructure'}):
                features['architecture_only'] = 1.0

    # ── NEW PENALTY: LANGCHAIN WRAPPER ONLY ──
    lc_hits = _fuzzy_match_count(LANGCHAIN_WRAPPER_SKILLS, skill_names)
    ret_hits = _fuzzy_match_count(RETRIEVAL_RANKING_SKILLS, skill_names)
    if lc_hits > 0 and ret_hits == 0:
        features['langchain_wrapper_only'] = 1.0
    else:
        features['langchain_wrapper_only'] = 0.0

    # ── 14-20. BEHAVIORAL SIGNALS ──
    rr_rate = signals.get('recruiter_response_rate', 0.0)
    if rr_rate < 0:
        rr_rate = 0.0
    features['recruiter_response_rate'] = rr_rate

    ic_rate = signals.get('interview_completion_rate', 0.0)
    if ic_rate < 0:
        ic_rate = 0.0
    features['interview_completion_rate'] = ic_rate

    features['open_to_work'] = 1.0 if signals.get('open_to_work_flag', False) else 0.0

    # Recency
    last_active = signals.get('last_active_date', '2025-01-01')
    try:
        dt = datetime.strptime(str(last_active)[:10], '%Y-%m-%d')
        days_inactive = max((datetime(2026, 6, 30) - dt).days, 0)
        features['recency'] = max(1.0 - (days_inactive / 180.0), 0.0)
    except:
        features['recency'] = 0.3

    # Notice period
    notice = signals.get('notice_period_days', 90)
    if notice <= 30:
        features['notice_fit'] = 1.0
    elif notice <= 60:
        features['notice_fit'] = 0.7
    elif notice <= 90:
        features['notice_fit'] = 0.4
    else:
        features['notice_fit'] = 0.2

    # GitHub activity
    github = signals.get('github_activity_score', -1)
    features['github_activity'] = max(github / 100.0, 0.0) if github >= 0 else 0.0

    # Saved by recruiters (social proof)
    saved = signals.get('saved_by_recruiters_30d', 0)
    features['recruiter_interest'] = min(saved / 20.0, 1.0)

    # ── 21. LOCATION FIT ──
    if any(loc in location for loc in GOOD_LOCATIONS):
        features['location_fit'] = 1.0
    elif signals.get('willing_to_relocate', False):
        features['location_fit'] = 0.6
    elif 'india' in profile.get('country', '').lower():
        features['location_fit'] = 0.4
    else:
        features['location_fit'] = 0.2

    # ── 22. EDUCATION RELEVANCE ──
    edu_score = 0.0
    for edu in education:
        field = edu.get('field_of_study', '').lower()
        tier = edu.get('tier', 'unknown')
        degree = edu.get('degree', '').lower()

        if any(f in field for f in CS_FIELDS):
            edu_score += 0.3
        if tier in ('tier_1', 'tier_2'):
            edu_score += 0.15
        if 'phd' in degree or 'ph.d' in degree:
            edu_score += 0.1
        elif 'master' in degree or 'm.tech' in degree or 'm.sc' in degree:
            edu_score += 0.05
    features['education_relevance'] = min(edu_score, 1.0)

    # ── 23. PROFILE COMPLETENESS ──
    completeness = signals.get('profile_completeness_score', 50)
    features['profile_completeness'] = completeness / 100.0

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# Weights derived directly from JD priority ordering
WEIGHTS = {
    # PRIMARY: What do they do? (66%)
    'retrieval_ranking_skills': 0.12,
    'career_ml_evidence':    0.12,  # Increased for prod deployments
    'title_relevance':       0.12,
    'must_have_skills':      0.10,
    'must_have_proficiency': 0.10,
    'evaluation_metrics':    0.05,
    'career_ml_titles':      0.05,

    # SECONDARY: Do they fit? (20%)
    'product_company':       0.10,  # Increased
    'experience_fit':        0.05,
    'production_skills':     0.03,
    'nice_to_have_skills':   0.01,
    'core_ml':               0.01,  # Decreased generic AI

    # TERTIARY: Are they available? (12%)
    'recruiter_response_rate': 0.03,
    'recency':                 0.02,
    'interview_completion_rate': 0.02,
    'career_stability':      0.02,
    'notice_fit':              0.01,
    'open_to_work':            0.01,
    'github_activity':         0.01,
    'recruiter_interest':      0.01,
    'location_fit':            0.01,

    # MINOR (2%)
    'education_relevance':   0.01,
    'profile_completeness':  0.01,
}

PENALTY_WEIGHTS = {
    'consulting_only':        0.40,
    'pure_research':          0.40,
    'keyword_stuffing':       0.30,
    'architecture_only':      0.20,
    'langchain_wrapper_only': 0.15,
    'wrong_domain':           0.20,
}


def compute_composite_score(features):
    """
    Compute the final composite score from extracted features.
    Returns a score roughly in range 0-1 (can exceed slightly).
    """
    # Positive score
    score = 0.0
    for fname, weight in WEIGHTS.items():
        score += features.get(fname, 0.0) * weight

    # Penalties (subtract)
    for fname, weight in PENALTY_WEIGHTS.items():
        score -= features.get(fname, 0.0) * weight

    # Hard ceiling for irrelevant titles
    if features.get('title_relevance', 0) == 0.0:
        score *= 0.25  # Irrelevant title = massive reduction

    return max(score, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: DYNAMIC REASONING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_reasoning(cand, features, score, rank):
    """
    Generate an explanation following the required template.
    """
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    title = profile.get('current_title', 'Unknown Role')
    yoe = profile.get('years_of_experience', 0)
    company = profile.get('current_company', 'Unknown Company')
    if not company:
        company = 'Unknown Company'
    
    # Get top relevant skills
    adv_skills = [s.get('name') for s in skills
                  if s.get('proficiency') in ('advanced', 'expert')][:5]
    skill_str = ", ".join(adv_skills[:3]) if adv_skills else "core engineering skills"

    # Determine match reason
    match_reasons = []
    if features.get('retrieval_ranking_skills', 0) > 0.6:
        match_reasons.append(f"expertise in {skill_str} and direct experience relevant to retrieval and ranking systems")
    elif features.get('must_have_skills', 0) > 0.4:
        match_reasons.append(f"expertise in {skill_str} and related experience relevant to retrieval and ranking systems")
    else:
        match_reasons.append(f"expertise in {skill_str} and transferable experience")
        
    match_str = match_reasons[0]

    # Additional strengths
    strengths = []
    if features.get('career_ml_evidence', 0) > 0.5:
        strengths.append("production ML deployment")
    if features.get('evaluation_metrics', 0) > 0.5:
        strengths.append("ranking evaluation")
    if features.get('product_company', 0) > 0.4:
        strengths.append("product company background")
        
    strengths_str = ""
    if strengths:
        if len(strengths) > 1:
            strengths_str = f" Additional strengths include {', '.join(strengths[:-1])} and {strengths[-1]}."
        else:
            strengths_str = f" Additional strengths include {strengths[0]}."

    # Concerns
    concerns = []
    notice = signals.get('notice_period_days', 90)
    if notice > 60:
        concerns.append(f"Extended Notice Period ({notice} days)")
    if yoe < 4:
        concerns.append("Junior Profile")
    elif yoe > 12:
        concerns.append("Over-experience")
    
    if features.get('must_have_skills', 0) < 0.3:
        concerns.append("Missing core skills")
    if features.get('consulting_only', 0) > 0.5:
        concerns.append("Consulting-only background")
    if features.get('pure_research', 0) > 0.5:
        concerns.append("Academic/Research only")
        
    concerns_str = ""
    if concerns:
        concerns_str = f" Concerns: {', '.join(concerns)}."

    reasoning = f"{title} with {yoe:.1f} years of experience, currently at {company}. Strong match due to {match_str}.{strengths_str}{concerns_str}"

    return reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def score_candidate(cand):
    """
    Full pipeline: filter → extract features → score.
    Returns (score, features, is_filtered, filter_reason).
    """
    # Stage 1: Hard filters
    is_hp, hp_reason = detect_honeypot(cand)
    if is_hp:
        return -1.0, {}, True, f"honeypot: {hp_reason}"

    is_disq, disq_reason = is_hard_disqualified(cand)
    if is_disq:
        return -1.0, {}, True, f"disqualified: {disq_reason}"

    # Stage 2: Feature extraction
    features = extract_rich_features(cand)

    # Stage 3: Composite score
    score = compute_composite_score(features)

    return score, features, False, ""
