import gradio as gr
import os
import subprocess
import shutil

def run_pipeline(candidate_file):
    # Determine the extension
    ext = os.path.splitext(candidate_file.name)[1]
    
    if ext == ".csv":
        target_path = "candidates_diverse_1000.csv"
    else:
        target_path = "candidates.jsonl"
        
    # Copy the uploaded file to the expected path in the root folder
    shutil.copy(candidate_file.name, target_path)
    
    # Run the scoring pipeline
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        result = subprocess.run(
            ["python3", "scoring.py"],
            cwd="src",
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None, f"Error running the pipeline:\n{e.stderr}"
        
    # Check if submission.csv was created
    submission_path = "submission.csv"
    if os.path.exists(submission_path):
        return submission_path, "Successfully generated submission.csv!"
    else:
        return None, "Pipeline finished but submission.csv was not found."

with gr.Blocks(title="Redrob Ranker Pipeline") as demo:
    gr.Markdown("# Redrob AI Challenge - Candidate Ranking")
    gr.Markdown("Upload your `candidates.jsonl` or `candidates_diverse_1000.csv` file to generate rankings.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Candidate Data (.jsonl or .csv)")
        
    with gr.Row():
        run_btn = gr.Button("Run Ranking Pipeline", variant="primary")
        
    with gr.Row():
        status_output = gr.Textbox(label="Status")
        file_output = gr.File(label="Download Submission CSV")
        
    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[file_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
