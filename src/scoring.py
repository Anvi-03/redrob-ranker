"""
scoring.py — Production-grade multi-stage candidate ranking pipeline.

Architecture:
  Stage 1: Semantic Retrieval (FAISS) — if pre-computed embeddings available
  Stage 2: Hard Filters (honeypots, disqualifiers)
  Stage 3: Rich Feature Extraction (25+ features)
  Stage 4: Composite Scoring (heuristic weights + optional LightGBM reranker)
  Stage 5: Dynamic Reasoning Generation

Usage:
  1. Place candidates.jsonl or candidates_diverse_1000.csv in the root folder
  2. (Optional) Run precompute_embeddings.py first for semantic retrieval
  3. Run: cd src && OMP_NUM_THREADS=1 python3 scoring.py

Output: submission.csv in the root folder
"""

import os
import sys
import json
import gzip
import re
import time
import pandas as pd
import numpy as np
import logging

# Add src to path for imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ranking_engine import (
    score_candidate, extract_rich_features, compute_composite_score,
    generate_reasoning, detect_honeypot, is_hard_disqualified, WEIGHTS
)

# ── Config ──
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
OUTPUT_FILE = os.path.join(ROOT_DIR, "submission.csv")

# Candidate file discovery
CANDIDATE_FILES = [
    os.path.join(ROOT_DIR, "candidates_diverse_1000.csv"),
    os.path.join(ROOT_DIR, "candidates.jsonl"),
    os.path.join(ROOT_DIR, "[PUB] India_runs_data_and_ai_challenge",
                 "India_runs_data_and_ai_challenge", "candidates.jsonl"),
]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ═══════════════════════════════════════════════════════════════════════════════
# CSV PARSING (for candidates_diverse_1000.csv format)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_csv_row(row):
    """Parse a flat CSV row into structured candidate dict."""
    cand = {'candidate_id': row['candidate_id']}
    cand['profile'] = {
        'summary': str(row.get('summary', '')),
        'years_of_experience': float(row.get('years_of_experience', 0)) if pd.notna(row.get('years_of_experience')) else 0.0,
        'current_title': str(row.get('current_title', '')),
        'current_company': str(row.get('current_company', '')) if pd.notna(row.get('current_company')) else '',
        'location': str(row.get('location', '')),
        'country': str(row.get('country', '')) if pd.notna(row.get('country')) else '',
        'current_industry': str(row.get('current_industry', '')) if pd.notna(row.get('current_industry')) else '',
    }

    # Parse career_history
    career = []
    career_str = str(row.get('career_history', ''))
    if pd.notna(career_str) and career_str and career_str != 'nan':
        for job in career_str.split(' || '):
            title = ""
            desc = ""
            duration = 0
            is_current = False
            company = ""
            industry = ""
            if ' @ ' in job:
                parts = job.split(' @ ', 1)
                title = parts[0].strip()
                rest = parts[1]
                if ' (' in rest:
                    cparts = rest.split(' (', 1)
                    company = cparts[0].strip()
                    meta = '(' + cparts[1]
                else:
                    meta = rest
                if '): ' in meta:
                    dparts = meta.split('): ', 1)
                    desc = dparts[1].strip()
                    meta = dparts[0]
                dur_match = re.search(r'(\d+)mo', meta)
                if dur_match:
                    duration = int(dur_match.group(1))
                if 'Present' in meta:
                    is_current = True
                # Extract industry from metadata if present
                ind_match = re.search(r'\d{4}-\d{2}-\d{2}.*?,\s*\d+mo,\s*([^,]+)', meta)
                if ind_match:
                    industry = ind_match.group(1).strip()
            career.append({
                'title': title, 'company': company, 'description': desc,
                'duration_months': duration, 'is_current': is_current,
                'industry': industry,
            })
    cand['career_history'] = career

    # Parse skills
    skills = []
    skills_str = str(row.get('skills', ''))
    if pd.notna(skills_str) and skills_str and skills_str != 'nan':
        for sk in skills_str.split(' || '):
            name = sk
            prof = 'beginner'
            dur = 0
            endorsements = 0
            if ' (' in sk:
                name = sk.split(' (')[0].strip()
                if 'advanced' in sk: prof = 'advanced'
                elif 'expert' in sk: prof = 'expert'
                elif 'intermediate' in sk: prof = 'intermediate'
                dur_match = re.search(r'(\d+)mo', sk)
                if dur_match:
                    dur = int(dur_match.group(1))
                end_match = re.search(r'(\d+)\s*endorsements?', sk)
                if end_match:
                    endorsements = int(end_match.group(1))
            skills.append({
                'name': name, 'proficiency': prof,
                'duration_months': dur, 'endorsements': endorsements,
            })
    cand['skills'] = skills

    # Parse education
    education = []
    edu_str = str(row.get('education', ''))
    if pd.notna(edu_str) and edu_str and edu_str != 'nan':
        for edu_entry in edu_str.split(' || '):
            degree = ""
            field = ""
            tier = "unknown"
            if ' in ' in edu_entry:
                eparts = edu_entry.split(' in ', 1)
                degree = eparts[0].strip()
                rest = eparts[1]
                if ' (' in rest:
                    field = rest.split(' (')[0].strip().split(',')[0].strip()
                else:
                    field = rest.strip().split(',')[0].strip()
            if 'tier_1' in edu_entry: tier = 'tier_1'
            elif 'tier_2' in edu_entry: tier = 'tier_2'
            elif 'tier_3' in edu_entry: tier = 'tier_3'
            education.append({
                'degree': degree, 'field_of_study': field, 'tier': tier,
                'institution': '',
            })
    cand['education'] = education

    # Parse redrob_signals
    cand['redrob_signals'] = {
        'recruiter_response_rate': float(row.get('recruiter_response_rate', 0)) if pd.notna(row.get('recruiter_response_rate')) else 0.0,
        'interview_completion_rate': float(row.get('interview_completion_rate', 0)) if pd.notna(row.get('interview_completion_rate')) else 0.0,
        'willing_to_relocate': str(row.get('willing_to_relocate', '')).lower() == 'true',
        'last_active_date': str(row.get('last_active_date', '')) if pd.notna(row.get('last_active_date')) else '',
        'open_to_work_flag': str(row.get('open_to_work_flag', '')).lower() == 'true',
        'notice_period_days': int(row.get('notice_period_days', 90)) if pd.notna(row.get('notice_period_days')) else 90,
        'github_activity_score': float(row.get('github_activity_score', -1)) if pd.notna(row.get('github_activity_score')) else -1,
        'saved_by_recruiters_30d': int(row.get('saved_by_recruiters_30d', 0)) if pd.notna(row.get('saved_by_recruiters_30d')) else 0,
        'profile_completeness_score': float(row.get('profile_completeness_score', 50)) if pd.notna(row.get('profile_completeness_score')) else 50,
    }

    return cand


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def find_candidate_file():
    for path in CANDIDATE_FILES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No candidate file found. Tried: {CANDIDATE_FILES}")


def yield_candidates(file_path):
    """Yield candidate dicts from either CSV or JSONL."""
    if file_path.endswith('.csv'):
        for chunk in pd.read_csv(file_path, chunksize=5000):
            for _, row in chunk.iterrows():
                yield parse_csv_row(row)
    else:
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC RETRIEVAL (FAISS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_semantic_artifacts():
    """Load pre-computed embeddings and FAISS index if available."""
    jd_path = os.path.join(ARTIFACTS_DIR, "jd_embedding.npy")
    ids_path = os.path.join(ARTIFACTS_DIR, "candidate_ids.npy")
    emb_path = os.path.join(ARTIFACTS_DIR, "candidate_embeddings.npy")
    faiss_path = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")

    if not all(os.path.exists(p) for p in [jd_path, ids_path, emb_path]):
        return None

    logging.info("Loading pre-computed semantic artifacts...")
    jd_embedding = np.load(jd_path)
    candidate_ids = np.load(ids_path, allow_pickle=True)
    candidate_embeddings = np.load(emb_path)

    faiss_index = None
    if os.path.exists(faiss_path):
        try:
            import faiss
            faiss_index = faiss.read_index(faiss_path)
            logging.info(f"FAISS index loaded with {faiss_index.ntotal} vectors")
        except ImportError:
            logging.warning("faiss not installed — will use numpy dot product fallback")

    return {
        'jd_embedding': jd_embedding,
        'candidate_ids': candidate_ids,
        'candidate_embeddings': candidate_embeddings,
        'faiss_index': faiss_index,
    }


def semantic_retrieval(semantic_data, top_k=1500):
    """Retrieve top-K candidates by semantic similarity to JD."""
    jd_emb = semantic_data['jd_embedding']
    cand_ids = semantic_data['candidate_ids']

    if semantic_data['faiss_index'] is not None:
        import faiss
        scores, indices = semantic_data['faiss_index'].search(jd_emb, top_k)
        results = {}
        for i, idx in enumerate(indices[0]):
            if idx < len(cand_ids):
                results[cand_ids[idx]] = float(scores[0][i])
        return results
    else:
        # Numpy fallback
        cand_emb = semantic_data['candidate_embeddings']
        similarities = np.dot(cand_emb, jd_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = {}
        for idx in top_indices:
            results[cand_ids[idx]] = float(similarities[idx])
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: LightGBM RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

def load_lgb_model():
    """Load trained LightGBM model if available."""
    model_path = os.path.join(ARTIFACTS_DIR, "reranker_model.txt")
    if os.path.exists(model_path):
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=model_path)
            logging.info(f"LightGBM reranker loaded from {model_path}")
            return model
        except Exception as e:
            logging.warning(f"Could not load LightGBM model: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    candidate_file = find_candidate_file()
    logging.info(f"Candidate file: {candidate_file}")

    # ── Try to load semantic artifacts ──
    semantic_data = load_semantic_artifacts()
    use_semantic = semantic_data is not None

    if use_semantic:
        logging.info("Semantic retrieval ENABLED (pre-computed embeddings found)")
    else:
        logging.info("Semantic retrieval DISABLED (no pre-computed embeddings)")
        logging.info("Running pure heuristic scoring over all candidates...")

    # ── Semantic pre-filtering ──
    semantic_scores = {}
    semantic_shortlist = None
    if use_semantic:
        logging.info("Stage 1: FAISS semantic retrieval (top 1500)...")
        semantic_scores = semantic_retrieval(semantic_data, top_k=1500)
        semantic_shortlist = set(semantic_scores.keys())
        logging.info(f"  Retrieved {len(semantic_shortlist)} candidates by semantic similarity")

    # ── Process candidates ──
    logging.info("Stage 2-3: Feature extraction and scoring...")

    scored_candidates = []
    filtered_count = 0
    total_count = 0

    for cand in yield_candidates(candidate_file):
        total_count += 1
        cid = cand.get('candidate_id', '')

        # If using semantic retrieval, skip candidates not in shortlist
        # BUT: also process all candidates with ML-related titles (safety net)
        if semantic_shortlist is not None:
            if cid not in semantic_shortlist:
                title = cand.get('profile', {}).get('current_title', '').lower()
                ml_words = {'ml', 'ai', 'machine learning', 'data scientist',
                            'nlp', 'search', 'ranking', 'recommendation',
                            'deep learning', 'applied scientist'}
                if not any(w in title for w in ml_words):
                    continue

        # Score the candidate
        score, features, is_filtered, reason = score_candidate(cand)

        if is_filtered:
            filtered_count += 1
            continue

        # Boost with semantic similarity if available
        if cid in semantic_scores:
            sem_sim = semantic_scores[cid]
            features['semantic_similarity'] = max(sem_sim, 0.0)
            # Add semantic similarity to the score (weighted at 15%)
            score += features['semantic_similarity'] * 0.15

        scored_candidates.append({
            'cand': cand,
            'score': score,
            'features': features,
        })

    logging.info(f"  Processed {total_count} candidates, filtered {filtered_count}")
    logging.info(f"  {len(scored_candidates)} candidates scored")

    # ── Sort and select top 100 ──
    logging.info("Stage 4: Ranking...")
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    top_100 = scored_candidates[:100]

    # ── Normalize scores to 0-1 range (monotonically non-increasing) ──
    if top_100:
        max_score = top_100[0]['score']
        min_score = top_100[-1]['score']
        score_range = max_score - min_score if max_score > min_score else 1.0

    # ── Generate submission ──
    logging.info("Stage 5: Generating reasoning and submission CSV...")
    submission_rows = []
    for rank, item in enumerate(top_100, start=1):
        cand = item['cand']
        raw_score = item['score']
        features = item['features']

        # Normalize score to 0-1 range
        normalized_score = round((raw_score - min_score) / score_range, 6) if score_range > 0 else 0.5

        # Generate reasoning
        reasoning = generate_reasoning(cand, features, raw_score, rank)

        submission_rows.append({
            'candidate_id': cand.get('candidate_id'),
            'rank': rank,
            'score': normalized_score,
            'reasoning': reasoning,
        })

    df = pd.DataFrame(submission_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start_time
    logging.info(f"\n✅ Generated {OUTPUT_FILE} with {len(df)} candidates in {elapsed:.1f}s")

    # ── Quick validation ──
    logging.info("\n── Top 10 Preview ──")
    for _, row in df.head(10).iterrows():
        logging.info(f"  Rank {row['rank']}: {row['candidate_id']} | Score: {row['score']:.4f}")
        logging.info(f"    {row['reasoning'][:120]}...")

    # Verify monotonic scores
    scores = df['score'].tolist()
    is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    if not is_monotonic:
        logging.warning("Scores are NOT monotonically non-increasing! Fixing...")
        # Force monotonic
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                scores[i] = scores[i-1]
        df['score'] = scores
        df.to_csv(OUTPUT_FILE, index=False)

    logging.info(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
