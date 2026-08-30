"""Promote the current challenger run to champion.

Run this after both gate tests pass:

    uv run python src/experiment/promote_challenger.py

The reigning champion is retagged `stage=archived`; the challenger is retagged
`stage=champion`. Requires exactly one run tagged `stage=champion` and exactly one
tagged `stage=challenger` (bootstrap the champion by hand in the MLflow UI first).
"""
import dagshub
import mlflow

from utils.mlflow_utils import (
    CHAMPION,
    CHALLENGER,
    get_run_by_stage,
    promote_challenger,
)

# initialize dagshub and mlflow
dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")

experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

champion = get_run_by_stage(CHAMPION, experiment_id)
challenger = get_run_by_stage(CHALLENGER, experiment_id)

print(f"current champion : {champion.info.run_id}  ({champion.info.run_name})")
print(f"challenger       : {challenger.info.run_id}  ({challenger.info.run_name})")

old_id, new_id = promote_challenger(experiment_id)
print(f"archived  {old_id}")
print(f"promoted  {new_id} -> {CHAMPION}")
