# Redrob Data & AI Challenge - Resume Predictor & Reranker

This repository contains the production-ready machine learning pipeline for the **Redrob Resume Predictor** challenge. The objective of this system is to rank 100,000 candidate profiles against a specific "Senior AI Engineer" job description within strict computational limits (under 5 minutes, single CPU, < 16GB RAM).

The system uses a highly optimized **LambdaMART (LightGBM) Learning-to-Rank** model, trained on synthetic labeled data that strictly adheres to the JD's requirements, catching "honeypots" and "keyword-stuffer" traps along the way.

---

## 🛠 Prerequisites & Environment Setup

This project uses standard data science libraries and runs purely locally. It requires Python 3.9+.

1. **Clone the repository and enter the directory:**
   ```bash
   cd resume-predictor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Core libraries: `pandas`, `numpy`, `lightgbm`, `scikit-learn`*

---

## 🚀 Running the Production Pipeline

The pipeline is split into three main scripts. Run them sequentially from the `src/` directory.

### Step 1: Generate the Training Dataset
This script scans the 100k candidates, applies strict heuristic rules derived directly from the Job Description, and generates a diverse dataset of 1,000 auto-labeled candidates (0.0 to 3.0 scale). It actively detects impossible profiles (honeypots) and gives them a score of `0.0`.

```bash
cd src
python3 generate_training_candidates.py
```
*Outputs: `data/training_candidates_labeled.csv`*

### Step 2: Train the Reranker Model
This script extracts numerical features from the labeled candidates and trains a `lambdarank` LightGBM model. It performs 5-fold cross-validation, reports NDCG metrics, and saves the final model artifacts.

*(Note: On macOS ARM / Apple Silicon, LightGBM OpenMP might crash silently. We use `OMP_NUM_THREADS=1` to ensure stability.)*

```bash
OMP_NUM_THREADS=1 python3 train_reranker.py
```
*Outputs:*
- *Model Weights: `artifacts/reranker_model.txt`*
- *Training Report: `artifacts/reranker_training_report.md`*

### Step 3: Score and Generate Submission
This script applies the trained LightGBM model across all 100,000 candidates.

**Important:** Before running this step, ensure that the `candidates.jsonl` file containing the 100k candidates is placed directly in the **root folder** of the project (`resume-predictor/candidates.jsonl`).

```bash
cd src
OMP_NUM_THREADS=1 python3 scoring.py
```
*Outputs: `submission.csv` (in the root directory)*

---

## 🧠 System Architecture & Methodology

### 1. Labeling Strategy (`generate_training_candidates.py`)
To train a supervised ranking model without manually labeling thousands of resumes, we built a strict heuristic labeler:
- **Perfect Fit (2.5 - 3.0):** Candidates with ML/AI titles, 5-9 years of experience, production-level retrieval/ranking skills (e.g., LlamaIndex, Elasticsearch, Milvus), and product company backgrounds.
- **Honeypots / Disqualified (0.0):** Candidates who claim 12 years of experience but only have 1 year of career history, or "keyword stuffers" who list AI skills but have irrelevant titles (e.g., HR Manager, Graphic Designer).

### 2. Feature Engineering (`features.py`)
Rather than heavy LLM API calls (which violate the constraint of running without network access), we extract dense signals:
- **Semantic Similarity:** Keyword density of ML, Tech, and Production terms matched against the candidate's career descriptions.
- **Experience Relevance:** Maps the candidate's YoE to the JD's "sweet spot" (5-9 years).
- **Career Trajectory:** Penalizes "title chasers" who hop jobs every few months.
- **Engagement Signals:** Aggregates recruiter response rates and platform recency.

### 3. Model Architecture (`train_reranker.py`)
We use **LightGBM** with the `lambdarank` objective.
- Unlike traditional regression, `lambdarank` optimizes directly for ranking metrics like **NDCG@10**.
- The model successfully learns that `semantic_similarity` and `experience_relevance` are the strongest predictors of a top-tier candidate.

### 4. High-Performance Scoring (`scoring.py`)
- Processes the 100,000-line JSONL file using a stream reader to keep RAM usage well under the 16GB limit.
- Total processing and scoring takes **< 60 seconds** on a standard CPU.
- Automatically generates customized reasoning for the Top 100 candidates based on their rank and extracted skill overlap.
