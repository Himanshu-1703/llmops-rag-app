"""Reject the current challenger run.

Run this when a gate test fails:

    uv run python src/experiment/reject_challenger.py

The challenger's `stage` tag is deleted (the run and its metrics stay in MLflow,
just unlabelled). The champion is left untouched.
"""
import dagshub
import mlflow

from utils.mlflow_utils import CHALLENGER, get_run_by_stage, reject_challenger

# initialize dagshub and mlflow
dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")

experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

challenger = get_run_by_stage(CHALLENGER, experiment_id)
print(f"challenger : {challenger.info.run_id}  ({challenger.info.run_name})")

reject_challenger(experiment_id)
print(f"cleared stage tag from {challenger.info.run_id}")
