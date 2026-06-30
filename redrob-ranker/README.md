# Redrob Ranker

AI-powered candidate ranking system for intelligent resume analysis and scoring.

## Overview

Redrob Ranker uses semantic embeddings, feature engineering, and machine learning to rank job candidates against a given job description. It includes honeypot detection for inconsistent profiles and a Streamlit sandbox for interactive exploration.

## Project Structure

```
redrob-ranker/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── config/
│   └── weights.yaml
├── src/
│   ├── precompute.py
│   ├── finetune_embedder.py
│   ├── embeddings.py
│   ├── features.py
│   ├── honeypot.py
│   ├── scoring.py
│   ├── reasoning.py
│   ├── rank.py
│   └── train_reranker.py
├── artifacts/
│   └── finetuned-embedder/
│       └── .gitkeep
├── data/
│   ├── sample_candidates.jsonl
│   └── job_description.json
├── tests/
│   ├── test_scoring.py
│   ├── test_honeypot.py
│   └── test_features.py
└── sandbox_app.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit sandbox
streamlit run sandbox_app.py

# Run tests
pytest tests/
```
