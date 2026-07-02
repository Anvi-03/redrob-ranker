---
title: Redrob Ranker
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.20.1"
app_file: app.py
pinned: false
---

# Redrob AI Challenge — Intelligent Candidate Discovery & Ranking

A **production-grade multi-stage ranking engine** that discovers and ranks the top 100 candidates from a 100K candidate pool against a Senior AI Engineer job description.

## 🏗 Architecture

```
Pre-computation (offline, one-time):
  sentence-transformers (all-MiniLM-L6-v2)
    → 100K candidate embeddings (384-dim)
    → FAISS IndexFlatIP for semantic retrieval

Ranking Pipeline (online, <60s on CPU):
  Stage 1: FAISS Semantic Retrieval → top 1500 by cosine similarity
  Stage 2: Hard Filters (honeypots, consulting-only, pure-research)
  Stage 3: Rich Feature Extraction (25+ features per candidate)
  Stage 4: Weighted Composite Scoring (JD-derived weights)
  Stage 5: Dynamic Reasoning Generation (per-candidate, non-templated)
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Heuristic scoring over LightGBM** | Training on self-labeled data causes train/serve skew. Direct scoring uses richer signals and is fully transparent. |
| **Sentence-transformer pre-computation** | Enables true semantic similarity (not keyword matching) while respecting the 5-min CPU runtime constraint. |
| **FAISS recall stage** | Reduces candidate pool from 100K → 1500 before expensive feature extraction, keeping runtime under budget. |
| **25+ features** | Each JD requirement mapped to a measurable signal: title relevance, must-have skills (proficiency-weighted), career ML evidence, product company experience, keyword stuffing detection, behavioral signals. |
| **Keyword stuffing penalty** | Explicitly penalizes profiles where claimed skills lack career evidence — the JD warns this is a trap. |

---

## 🛠 Setup

```bash
# Clone and enter directory
cd redrob-ranker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running

### Option A: Full Pipeline (Recommended)

**Step 1: Place candidate data in root folder**
Put `candidates.jsonl` or `candidates_diverse_1000.csv` in the root directory.

**Step 2: Pre-compute embeddings (one-time, ~15-30 min on CPU)**
```bash
cd src
python3 precompute_embeddings.py
```
This generates `artifacts/faiss_index.bin`, `artifacts/candidate_embeddings.npy`, etc.

**Step 3: Generate submission**
```bash
cd src
OMP_NUM_THREADS=1 python3 scoring.py
```
Output: `submission.csv` in root directory.

### Option B: Heuristic-Only (No Pre-computation)

If you skip the embedding step, the pipeline automatically falls back to pure heuristic scoring over all 100K candidates. This still produces good rankings but without the semantic retrieval stage.

```bash
cd src
OMP_NUM_THREADS=1 python3 scoring.py
```

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── submission.csv                    # Final output
├── candidates_diverse_1000.csv       # Input dataset (100K candidates)
├── artifacts/
│   ├── reranker_model.txt            # LightGBM model (optional)
│   ├── faiss_index.bin               # FAISS vector index (pre-computed)
│   ├── candidate_embeddings.npy      # Dense embeddings (pre-computed)
│   ├── candidate_ids.npy             # Candidate ID mapping
│   └── jd_embedding.npy             # JD embedding
├── src/
│   ├── scoring.py                    # Main entry point — multi-stage pipeline
│   ├── ranking_engine.py             # Core: features, scoring, reasoning
│   ├── precompute_embeddings.py      # Pre-computation: embeddings + FAISS
│   ├── train_reranker.py             # LightGBM training (optional)
│   ├── generate_training_candidates.py  # Training data generation
│   └── features.py                   # Legacy feature extraction
└── data/
    └── training_candidates_labeled.csv  # Labeled training data
```

---

## 🧠 Feature Engineering (25+ Features)

| Category | Features | Weight |
|---|---|---|
| **Role Match** | `title_relevance`, `career_ml_titles` | 23% |
| **Skill Match** | `must_have_skills`, `must_have_proficiency`, `core_ml`, `nice_to_have_skills`, `production_skills` | 39% |
| **Career Quality** | `career_ml_evidence`, `product_company`, `career_stability` | 17% |
| **Experience** | `experience_fit` | 6% |
| **Behavioral** | `recruiter_response_rate`, `recency`, `interview_completion_rate`, `notice_fit`, `open_to_work`, `github_activity`, `recruiter_interest`, `location_fit` | 12% |
| **Education** | `education_relevance`, `profile_completeness` | 3% |
| **Penalties** | `keyword_stuffing` (-30%), `wrong_domain` (-15%) | Subtractive |

---

## 🛡 Anti-Gaming Measures

1. **Honeypot Detection**: Expert skills with 0 months used, impossible job tenures, career/YoE mismatches
2. **Keyword Stuffing Penalty**: High skill count relative to YoE, skills with no career evidence, irrelevant title + ML skills
3. **Wrong Domain Penalty**: CV/speech/robotics skills when JD needs NLP/retrieval
4. **Title Chaser Detection**: Average tenure < 18 months across 3+ jobs
5. **Consulting-Only Filter**: Pure TCS/Infosys/Wipro careers with no product exposure
