#!/usr/bin/env python3
"""
Generate a diverse, auto-labeled training dataset of 1000 candidates.

STRICT labeling scheme (0-3):
  3.0  = PERFECT fit — ML/AI title, 5-9yr, retrieval/ranking/LLM production skills,
          product-company career, strong behavioral signals. Very rare.
  2.0  = Good fit — relevant adjacent title OR strong ML skills but missing some requirements
  1.0  = Marginal — some overlap but significant gaps
  0.0  = Irrelevant, honeypot, consulting-only, or completely wrong domain

The JD is for: Senior AI Engineer at Redrob (Series A)
  MUST HAVE: embeddings, retrieval, ranking, LLMs, vector DBs, strong Python, eval frameworks
  NICE TO HAVE: fine-tuning (LoRA/QLoRA), learning-to-rank, HR-tech
  SWEET SPOT: 5-9 yrs, product company, shipped ranking/search/recommendation to real users
  DISQUALIFIERS: only-consulting, only-research, only-CV/speech/robotics, title-chasers
"""

import json
import gzip
import random
import pandas as pd
import os
from datetime import datetime

random.seed(42)

CANDIDATES_FILE = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUT_FILE = "../data/training_candidates_labeled.csv"

os.makedirs("../data", exist_ok=True)
open_func = gzip.open if CANDIDATES_FILE.endswith('.gz') else open

# ── Skill taxonomy derived strictly from the JD ───────────────────────────

# These are what the JD literally says you MUST have
MUST_HAVE_SKILLS = {
    'embeddings', 'embedding', 'sentence-transformers', 'sentence transformers',
    'retrieval', 'rag', 'retrieval-augmented generation', 'semantic search',
    'hybrid search', 'bge', 'e5', 'vector search', 'bm25',
    'ranking', 'learning to rank', 're-ranking',
    'llm', 'llms', 'large language models',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'elasticsearch', 'faiss', 'chroma', 'chromadb', 'pgvector',
    'ndcg', 'mrr', 'map', 'a/b testing',
}

# Nice to have per JD
NICE_TO_HAVE_SKILLS = {
    'fine-tuning', 'fine-tuning llms', 'lora', 'qlora', 'peft',
    'xgboost', 'lightgbm', 'recommendation', 'recommendation systems',
    'langchain', 'llamaindex', 'haystack',
    'distributed systems', 'inference optimization',
}

# Core ML (general foundation)
CORE_ML = {
    'machine learning', 'deep learning', 'nlp', 'natural language processing',
    'transformers', 'neural networks', 'pytorch', 'tensorflow',
    'information retrieval', 'data science',
}

# Production engineering
PRODUCTION_SKILLS = {
    'python', 'docker', 'kubernetes', 'mlops', 'mlflow',
    'fastapi', 'flask', 'aws', 'gcp', 'azure', 'ci/cd',
    'deployment', 'production',
}

# Wrong-domain skills (JD explicitly says NOT these)
WRONG_DOMAIN = {
    'computer vision', 'image classification', 'object detection',
    'speech recognition', 'tts', 'text-to-speech', 'robotics',
    'autonomous driving', 'gans', 'image segmentation',
    'yolo', 'opencv', 'cnn', 'diffusion models',
}

# Consulting firms — JD explicitly disqualifies consulting-only careers
CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree', 'mphasis',
}

# Known product companies (signal for quality career)
PRODUCT_COMPANIES = {
    'razorpay', 'swiggy', 'zomato', 'flipkart', 'cred', 'ola',
    'paytm', 'phonepe', 'dream11', 'meesho', 'byju', 'unacademy',
    'freshworks', 'zoho', 'postman', 'haptik', 'rephrase.ai', 'sarvam ai',
    'google', 'microsoft', 'amazon', 'meta', 'apple',
    'uber', 'netflix', 'airbnb', 'stripe', 'shopify',
    'stark industries', 'wayne enterprises', 'pied piper', 'hooli',
    'acme corp', 'initech', 'globex inc', 'dunder mifflin',
}

GOOD_LOCATIONS = {'pune', 'noida', 'delhi', 'ncr', 'mumbai', 'hyderabad',
                  'bangalore', 'bengaluru', 'gurgaon', 'gurugram'}

# Titles that actually match the JD
PERFECT_TITLES = {
    'ai engineer', 'senior ai engineer', 'ml engineer',
    'senior ml engineer', 'machine learning engineer',
    'senior machine learning engineer', 'applied scientist',
    'nlp engineer', 'senior nlp engineer', 'search engineer',
    'ranking engineer', 'recommendation engineer',
}

GOOD_TITLES = {
    'data scientist', 'senior data scientist', 'research engineer',
    'ai research engineer', 'ai specialist', 'ml specialist',
    'junior ml engineer', 'senior software engineer (ml)',
}

ADJACENT_TITLES = {
    'software engineer', 'senior software engineer', 'backend engineer',
    'senior backend engineer', 'full stack developer', 'data engineer',
    'senior data engineer', 'platform engineer', 'tech lead',
}

# Completely irrelevant titles
IRRELEVANT_TITLES = {
    'marketing manager', 'hr manager', 'sales executive', 'content writer',
    'graphic designer', 'accountant', 'customer support', 'operations manager',
    'civil engineer', 'mechanical engineer', 'project manager',
    'business analyst',
}


# ── Honeypot detection ─────────────────────────────────────────────────────

def detect_honeypot(cand):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    yoe = profile.get('years_of_experience', 0)

    # 1. Expert proficiency with 0 months used
    expert_zero = [s['name'] for s in skills
                   if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0]
    if len(expert_zero) >= 3:
        return True, f"expert_zero_months({len(expert_zero)} skills)"

    # 2. Single job duration > total YoE
    yoe_months = yoe * 12
    for job in career:
        dur = job.get('duration_months', 0)
        if dur > yoe_months + 12:
            return True, f"impossible_tenure({job.get('title')}@{job.get('company')} {dur}m vs {yoe}yr)"

    # 3. Career/YoE ratio wildly off
    total_career = sum(j.get('duration_months', 0) for j in career)
    if total_career > 0 and yoe > 0:
        ratio = total_career / (yoe * 12)
        if ratio > 2.5 or ratio < 0.3:
            return True, f"career_yoe_mismatch(ratio={ratio:.2f})"

    # 4. Expert in 10+ skills with < 3 yrs
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count >= 10 and yoe < 3:
        return True, f"impossible_expertise({expert_count} expert skills, {yoe}yr)"

    return False, ""


# ── Strict label computation ───────────────────────────────────────────────

def compute_label(cand):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    signals = cand.get('redrob_signals', {})

    # ── Honeypot check ──
    is_hp, hp_reason = detect_honeypot(cand)
    if is_hp:
        return 0.0, f"honeypot: {hp_reason}"

    yoe = profile.get('years_of_experience', 0)
    title = profile.get('current_title', '').lower().strip()
    summary = profile.get('summary', '').lower()
    location = profile.get('location', '').lower()

    # Build text blob from summary + career descriptions
    text_blob = summary
    for job in career:
        text_blob += " " + job.get('title', '').lower()
        text_blob += " " + job.get('description', '').lower()

    cand_skill_names = {s.get('name', '').lower() for s in skills}
    cand_adv_expert = {s.get('name', '').lower() for s in skills
                       if s.get('proficiency') in ('advanced', 'expert')}

    # ── Skill matching (fuzzy) ──
    def count_matches(target_set, source):
        hits = 0
        for t in target_set:
            for s in source:
                if t in s or s in t:
                    hits += 1
                    break
        return hits

    must_have_skill = count_matches(MUST_HAVE_SKILLS, cand_skill_names)
    must_have_text = count_matches(MUST_HAVE_SKILLS, {text_blob})
    nice_to_have_skill = count_matches(NICE_TO_HAVE_SKILLS, cand_skill_names)
    core_ml_skill = count_matches(CORE_ML, cand_skill_names)
    core_ml_text = count_matches(CORE_ML, {text_blob})
    prod_skill = count_matches(PRODUCTION_SKILLS, cand_skill_names)
    wrong_domain_skill = count_matches(WRONG_DOMAIN, cand_skill_names)

    total_must_have = must_have_skill + must_have_text
    total_core_ml = core_ml_skill + core_ml_text
    total_nice = nice_to_have_skill

    # ── Title classification ──
    title_tier = 0  # 0=irrelevant, 1=adjacent, 2=good, 3=perfect
    for t in PERFECT_TITLES:
        if t == title or title.startswith(t) or title.endswith(t):
            title_tier = 3
            break
    if title_tier == 0:
        for t in GOOD_TITLES:
            if t == title or title.startswith(t) or title.endswith(t):
                title_tier = 2
                break
    if title_tier == 0:
        for t in ADJACENT_TITLES:
            if t == title or title.startswith(t) or title.endswith(t):
                title_tier = 1
                break
    # everything else stays 0 (irrelevant)

    # ── Career analysis ──
    has_product_company = False
    all_consulting = True
    prod_ml_in_career = 0  # count of production ML keywords in career descriptions

    career_prod_keywords = ['production', 'deployed', 'shipped', 'scale', 'ranking',
                            'search', 'retrieval', 'embedding', 'recommendation',
                            'inference', 'model serving', 'real users', 'pipeline',
                            'vector', 'llm', 'fine-tun']

    if career:
        for job in career:
            comp = job.get('company', '').lower()
            is_consulting = any(cf in comp for cf in CONSULTING_FIRMS)
            if not is_consulting:
                all_consulting = False
            if any(pc in comp for pc in PRODUCT_COMPANIES):
                has_product_company = True

            desc = job.get('description', '').lower()
            prod_ml_in_career += sum(1 for kw in career_prod_keywords if kw in desc)
    else:
        all_consulting = False  # no career = unknown, not consulting

    # Title-chaser check
    non_current_durs = [j.get('duration_months', 0) for j in career if not j.get('is_current', False)]
    title_chaser = len(non_current_durs) >= 3 and sum(non_current_durs) / len(non_current_durs) < 18

    # ── Behavioral signals ──
    rr_rate = signals.get('recruiter_response_rate', 0.0)
    ic_rate = signals.get('interview_completion_rate', 0.0)
    open_to_work = signals.get('open_to_work_flag', False)
    github_score = signals.get('github_activity_score', -1)

    last_active = signals.get('last_active_date', '2025-01-01')
    try:
        dt = datetime.strptime(last_active, '%Y-%m-%d')
        days_inactive = (datetime(2026, 6, 30) - dt).days
    except:
        days_inactive = 365

    loc_match = any(loc in location for loc in GOOD_LOCATIONS)
    willing_relocate = signals.get('willing_to_relocate', False)
    notice_days = signals.get('notice_period_days', 90)

    # ═══════════════════════════════════════════════════════════════════════
    # STRICT SCORING — Label 3 is VERY hard to get
    # ═══════════════════════════════════════════════════════════════════════

    # ── Hard disqualifiers → label 0 ──
    if all_consulting and len(career) >= 2:
        # Purely consulting career
        base = 0.1 + min(total_must_have * 0.05, 0.2) + min(total_core_ml * 0.03, 0.1)
        return round(min(base, 0.5), 3), "consulting_only"

    if title_tier == 0 and total_must_have == 0 and total_core_ml == 0:
        # Completely irrelevant — wrong title AND no ML skills at all
        base = 0.1 + (0.05 if has_product_company else 0)
        return round(min(base, 0.3), 3), "irrelevant_no_ml"

    # ── Score components ──

    # TITLE (0-1.0) — This is the biggest gate
    if title_tier == 3:
        title_score = 1.0
    elif title_tier == 2:
        title_score = 0.7
    elif title_tier == 1:
        title_score = 0.4
    else:
        title_score = 0.0  # Irrelevant title = hard ceiling

    # MUST-HAVE SKILLS (0-1.0) — The JD's core requirements
    must_have_score = min(total_must_have / 5.0, 1.0)

    # CORE ML foundation (0-0.5)
    core_ml_score = min(total_core_ml / 4.0, 0.5)

    # NICE-TO-HAVE (0-0.3)
    nice_score = min(total_nice / 3.0, 0.3)

    # EXPERIENCE FIT (0-0.6)
    if 5 <= yoe <= 9:
        exp_score = 0.6
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        exp_score = 0.4
    elif 3 <= yoe < 4:
        exp_score = 0.25
    else:
        exp_score = 0.1

    # CAREER QUALITY (0-0.6)
    career_score = 0.0
    if has_product_company:
        career_score += 0.2
    if prod_ml_in_career >= 5:
        career_score += 0.25
    elif prod_ml_in_career >= 2:
        career_score += 0.1
    if not title_chaser:
        career_score += 0.1
    if all_consulting:
        career_score = 0.0
    career_score = min(career_score, 0.6)

    # BEHAVIORAL (0-0.3) — secondary signal
    behav_score = 0.0
    behav_score += rr_rate * 0.08
    behav_score += ic_rate * 0.04
    if open_to_work:
        behav_score += 0.03
    if days_inactive < 30:
        behav_score += 0.04
    elif days_inactive < 90:
        behav_score += 0.02
    if github_score > 30:
        behav_score += 0.04
    if loc_match:
        behav_score += 0.04
    elif willing_relocate:
        behav_score += 0.02
    if notice_days <= 30:
        behav_score += 0.03
    behav_score = min(behav_score, 0.3)

    # ── Combine ──
    raw = title_score + must_have_score + core_ml_score + nice_score + exp_score + career_score + behav_score
    # Max possible: 1.0 + 1.0 + 0.5 + 0.3 + 0.6 + 0.6 + 0.3 = 4.3

    # ── Penalties ──
    penalty = 1.0

    # Wrong domain dominant (CV/speech when JD wants NLP/retrieval)
    if wrong_domain_skill > total_must_have + total_core_ml and wrong_domain_skill >= 3:
        penalty *= 0.5

    # Title chaser
    if title_chaser:
        penalty *= 0.7

    # Inactive > 6 months
    if days_inactive > 180:
        penalty *= 0.6

    # Irrelevant title is a HARD ceiling — max label ~1.5 no matter what skills
    if title_tier == 0:
        penalty *= 0.35

    raw *= penalty

    # Scale to 0-3 (a "perfect" candidate scores ~4.0+ raw before scaling)
    label = min(raw * 3.0 / 4.0, 3.0)

    # ── FINAL GATE for label 3: must pass ALL of these ──
    if label >= 2.5:
        passes_gate = (
            title_tier >= 2 and           # Must have ML/AI/DS title
            total_must_have >= 2 and      # Must have ≥2 JD must-have skills
            5 <= yoe <= 12 and            # Must be in experience range
            has_product_company and       # Must have product company in career
            not all_consulting and        # Must not be consulting-only
            days_inactive < 120           # Must be recently active
        )
        if not passes_gate:
            label = min(label, 2.3)  # Cap at 2.3 if gate fails

    if label >= 2.8:
        passes_elite = (
            title_tier >= 3 and           # Must have PERFECT title (ML/AI Engineer)
            total_must_have >= 3 and      # Must have ≥3 JD must-have skills
            5 <= yoe <= 9 and             # Sweet spot experience
            prod_ml_in_career >= 3 and    # Career shows production ML work
            rr_rate >= 0.3                # Responsive to recruiters
        )
        if not passes_elite:
            label = min(label, 2.75)

    # Determine category for debugging
    if title_chaser:
        category = "title_chaser"
    elif days_inactive > 180:
        category = "inactive"
    elif wrong_domain_skill > total_must_have + total_core_ml and wrong_domain_skill >= 3:
        category = "cv_speech_dominant"
    elif title_tier == 0:
        category = "irrelevant_title"
    else:
        category = "normal"

    return round(max(label, 0.0), 3), category


# ═══════════════════════════════════════════════════════════════════════════
# Main: Process all 100K, bucket, and sample
# ═══════════════════════════════════════════════════════════════════════════

print("Reading all 100K candidates with STRICT labeling...")

buckets = {
    'tier3': [],    # 2.5-3.0
    'tier2': [],    # 1.5-2.5
    'tier1': [],    # 0.5-1.5
    'tier0': [],    # 0.0-0.5
    'honeypot': [],
}

all_categories = {}

with open_func(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        cand = json.loads(line.strip())
        label, category = compute_label(cand)

        entry = (cand, label, category)

        if category.startswith('honeypot'):
            buckets['honeypot'].append(entry)
        elif label >= 2.5:
            buckets['tier3'].append(entry)
        elif label >= 1.5:
            buckets['tier2'].append(entry)
        elif label >= 0.5:
            buckets['tier1'].append(entry)
        else:
            buckets['tier0'].append(entry)

        all_categories[category] = all_categories.get(category, 0) + 1

print("\n── Category distribution across 100K ──")
for cat, count in sorted(all_categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print(f"\n── Bucket sizes ──")
for bname, entries in buckets.items():
    if entries:
        labels = [e[1] for e in entries]
        print(f"  {bname}: {len(entries)} (label {min(labels):.3f} - {max(labels):.3f})")
    else:
        print(f"  {bname}: 0")

# ── Diverse sampling ──
TOTAL = 1000
sample = []

# All honeypots
sample.extend(buckets['honeypot'])
print(f"\nTaking {len(buckets['honeypot'])} honeypots")

# Tier 3 — take up to 50 (these should be rare and genuinely excellent)
t3 = min(50, len(buckets['tier3']))
t3_s = random.sample(buckets['tier3'], t3) if len(buckets['tier3']) > t3 else buckets['tier3']
sample.extend(t3_s)
print(f"Taking {len(t3_s)} tier-3 (best)")

# Tier 2 — ~200
t2 = min(200, len(buckets['tier2']))
t2_s = random.sample(buckets['tier2'], t2) if len(buckets['tier2']) > t2 else buckets['tier2']
sample.extend(t2_s)
print(f"Taking {len(t2_s)} tier-2 (good)")

# Tier 1 — ~350
t1 = min(350, len(buckets['tier1']))
t1_s = random.sample(buckets['tier1'], t1) if len(buckets['tier1']) > t1 else buckets['tier1']
sample.extend(t1_s)
print(f"Taking {len(t1_s)} tier-1 (marginal)")

# Tier 0 — fill rest
t0 = TOTAL - len(sample)
t0_s = random.sample(buckets['tier0'], t0) if len(buckets['tier0']) > t0 else buckets['tier0']
sample.extend(t0_s)
print(f"Taking {len(t0_s)} tier-0 (worst)")

random.shuffle(sample)

# ── Build CSV ──
rows = []
for cand, label, category in sample:
    prof = cand.get('profile', {})
    skills_list = cand.get('skills', [])
    career_list = cand.get('career_history', [])

    top_skills = ", ".join([s.get('name') for s in skills_list
                           if s.get('proficiency') in ('advanced', 'expert')][:10])
    career_str = " | ".join([
        f"{j.get('title')} @ {j.get('company')} ({j.get('duration_months')}m)"
        for j in career_list[:3]
    ])

    rows.append({
        'candidate_id': cand.get('candidate_id'),
        'label': label,
        'current_title': prof.get('current_title', ''),
        'years_of_experience': prof.get('years_of_experience', 0),
        'top_skills': top_skills,
        'career_history': career_str,
        'summary': prof.get('summary', ''),
        'category': category,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)

print(f"\n✅ Exported {len(df)} candidates to {OUT_FILE}")
print(f"\n── Final label distribution ──")
print(f"  Label 2.5-3.0 (PERFECT):   {len(df[df['label'] >= 2.5])}")
print(f"  Label 1.5-2.5 (good):      {len(df[(df['label'] >= 1.5) & (df['label'] < 2.5)])}")
print(f"  Label 0.5-1.5 (marginal):  {len(df[(df['label'] >= 0.5) & (df['label'] < 1.5)])}")
print(f"  Label 0.0-0.5 (worst):     {len(df[df['label'] < 0.5])}")
print(f"  Label == 0.0 (honeypots):  {len(df[df['label'] == 0.0])}")

# Show top-10 to verify quality
print(f"\n── Top 10 candidates (should be ML/AI engineers with retrieval/ranking skills) ──")
top10 = df.nlargest(10, 'label')
for _, r in top10.iterrows():
    print(f"  {r['candidate_id']} | {r['label']:.3f} | {r['current_title']} | {r['years_of_experience']}yr | {str(r['top_skills'])[:70]}")
