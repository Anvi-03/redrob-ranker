"""
precompute_embeddings.py — Pre-computation step (runs once, offline).

Uses sentence-transformers to encode the JD and all 100K candidates into
dense embeddings, then builds a FAISS index for fast semantic retrieval.

This step can take 15-30 minutes on CPU. The resulting artifacts are saved
to disk and loaded at ranking time for instant semantic similarity lookup.

Output artifacts:
  - artifacts/candidate_embeddings.npy  (100K x 384 float32)
  - artifacts/candidate_ids.npy         (100K string IDs)
  - artifacts/jd_embedding.npy          (1 x 384 float32)
  - artifacts/faiss_index.bin           (FAISS IndexFlatIP)
"""

import os
import sys
import json
import gzip
import re
import numpy as np
import pandas as pd
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Try multiple candidate file locations
CANDIDATE_FILES = [
    os.path.join(ROOT_DIR, "candidates_diverse_1000.csv"),
    os.path.join(ROOT_DIR, "candidates.jsonl"),
    os.path.join(ROOT_DIR, "[PUB] India_runs_data_and_ai_challenge",
                 "India_runs_data_and_ai_challenge", "candidates.jsonl"),
]

# The JD text — condensed version for embedding
JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI (Series A).
Requirements: Production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5). Production experience
with vector databases or hybrid search infrastructure (Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). Strong Python.
Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP).
5-9 years experience. Product company background preferred.
Nice to have: LLM fine-tuning (LoRA, QLoRA), learning-to-rank,
recommendation systems, distributed systems, HR-tech exposure.
Must have shipped ranking, search, or recommendation systems to real users.
"""


def find_candidate_file():
    for path in CANDIDATE_FILES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No candidate file found. Tried: {CANDIDATE_FILES}")


def parse_csv_row(row):
    """Parse a flat CSV row into structured dict (same as scoring.py)."""
    cand = {'candidate_id': row['candidate_id']}
    cand['profile'] = {
        'summary': str(row.get('summary', '')),
        'years_of_experience': float(row.get('years_of_experience', 0)) if pd.notna(row.get('years_of_experience')) else 0.0,
        'current_title': str(row.get('current_title', '')),
        'location': str(row.get('location', '')),
        'current_company': str(row.get('current_company', '')) if pd.notna(row.get('current_company')) else '',
    }

    career = []
    career_str = str(row.get('career_history', ''))
    if pd.notna(career_str) and career_str:
        for job in career_str.split(' || '):
            title = ""
            desc = ""
            duration = 0
            company = ""
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
            career.append({'title': title, 'company': company, 'description': desc, 'duration_months': duration})
    cand['career_history'] = career

    skills = []
    skills_str = str(row.get('skills', ''))
    if pd.notna(skills_str) and skills_str:
        for sk in skills_str.split(' || '):
            name = sk
            prof = 'beginner'
            dur = 0
            if ' (' in sk:
                name = sk.split(' (')[0].strip()
                if 'advanced' in sk: prof = 'advanced'
                elif 'expert' in sk: prof = 'expert'
                elif 'intermediate' in sk: prof = 'intermediate'
                dur_match = re.search(r'(\d+)mo', sk)
                if dur_match:
                    dur = int(dur_match.group(1))
            skills.append({'name': name, 'proficiency': prof, 'duration_months': dur})
    cand['skills'] = skills

    return cand


def candidate_to_text(cand):
    """Convert candidate to a concise text string for embedding."""
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])

    parts = []

    # Title
    title = profile.get('current_title', '')
    if title:
        parts.append(f"Title: {title}")

    # Summary (first 300 chars)
    summary = profile.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary[:300]}")

    # Top skills
    top_skills = [s.get('name', '') for s in skills
                  if s.get('proficiency') in ('advanced', 'expert')][:8]
    if top_skills:
        parts.append(f"Key skills: {', '.join(top_skills)}")

    # Recent career
    for job in career[:3]:
        jtitle = job.get('title', '')
        jcomp = job.get('company', '')
        jdesc = job.get('description', '')[:150]
        if jtitle:
            job_str = f"{jtitle}"
            if jcomp:
                job_str += f" at {jcomp}"
            if jdesc:
                job_str += f": {jdesc}"
            parts.append(job_str)

    return ". ".join(parts)


def yield_candidates(file_path):
    """Yield (candidate_id, candidate_dict) from either CSV or JSONL."""
    if file_path.endswith('.csv'):
        for chunk in pd.read_csv(file_path, chunksize=5000):
            for _, row in chunk.iterrows():
                cand = parse_csv_row(row)
                yield cand['candidate_id'], cand
    else:
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                cand = json.loads(line.strip())
                yield cand['candidate_id'], cand


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    candidate_file = find_candidate_file()
    print(f"Using candidate file: {candidate_file}")

    # ── Load sentence-transformers ──
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ── Encode JD ──
    print("Encoding job description...")
    jd_embedding = model.encode([JD_TEXT], normalize_embeddings=True, show_progress_bar=False)
    np.save(os.path.join(ARTIFACTS_DIR, "jd_embedding.npy"), jd_embedding)
    print(f"JD embedding shape: {jd_embedding.shape}")

    # ── Encode all candidates in batches ──
    print(f"\nEncoding all candidates from {candidate_file}...")
    start_time = time.time()

    all_ids = []
    all_texts = []

    for cid, cand in yield_candidates(candidate_file):
        all_ids.append(cid)
        all_texts.append(candidate_to_text(cand))

    total = len(all_ids)
    print(f"Loaded {total} candidates. Starting batch encoding...")

    # Encode in batches of 512
    batch_size = 512
    all_embeddings = []
    for i in range(0, total, batch_size):
        batch = all_texts[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)
        done = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  Encoded {done}/{total} ({done*100/total:.1f}%) — {rate:.0f} cand/s — ETA: {eta:.0f}s", end='\r')

    print()

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    elapsed = time.time() - start_time
    print(f"Encoding complete in {elapsed:.1f}s. Shape: {embeddings.shape}")

    # Save embeddings and IDs
    np.save(os.path.join(ARTIFACTS_DIR, "candidate_embeddings.npy"), embeddings)
    np.save(os.path.join(ARTIFACTS_DIR, "candidate_ids.npy"), np.array(all_ids))
    print(f"Saved embeddings and IDs to {ARTIFACTS_DIR}")

    # ── Build FAISS index ──
    print("Building FAISS index...")
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "faiss_index.bin"))
        print(f"FAISS index built with {index.ntotal} vectors")
    except ImportError:
        print("WARNING: faiss not installed. Skipping FAISS index.")
        print("The ranking will still work using numpy-based similarity.")
        print("To install: pip install faiss-cpu")

    total_time = time.time() - start_time
    print(f"\n✅ Pre-computation complete in {total_time:.1f}s")
    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
