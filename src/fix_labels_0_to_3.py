import pandas as pd
import json, gzip

# Read sample_submission.csv
sample_csv = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_submission.csv"
df_sample = pd.read_csv(sample_csv)

# Create a mapping of candidate_id to a 0-3 score
id_to_score = {}
for _, row in df_sample.iterrows():
    cid = row['candidate_id']
    score = float(row['score'])
    
    # Scale from 0-1 to 0-3
    scaled_score = score * 3.0
    id_to_score[cid] = f"{scaled_score:.3f}"

# Update the CSV
df_out = pd.read_csv("../data/labeled_candidates.csv")
df_out['label'] = df_out['candidate_id'].map(id_to_score)
df_out.to_csv("../data/labeled_candidates.csv", index=False)
print("Updated labels to 0-3 scale.")
