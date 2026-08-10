import pytest
from pathlib import Path
import json
import mlflow
import dagshub
from utils.mlflow_utils import get_run_info, get_metrics_from_stage
from typing import Literal

ROOT_DIR = Path(__file__).parent.parent
historical_runs_json_path = ROOT_DIR / "historical_runs.json"
threshold_values_path = ROOT_DIR / "thresholds.json"

TARGET_STAGE = "staging"
MULTIPLIER = 2


def get_latest_runid(json_path: Path | str) -> str:
    if isinstance(json_path, str):
        json_path = Path(json_path)
    
    if json_path.exists():    
        with open(json_path, "r") as file:
            historical_runs = json.load(file)
            latest_run_id = historical_runs[-1]["run_id"]
            
            return latest_run_id


def load_thresholds(thresholds_type: Literal["noise_thresholds", "historical_thresholds"], thresholds_path: Path | str) -> dict:
    if isinstance(thresholds_path, str):
            thresholds_path= Path(thresholds_path)
        
    if thresholds_path.exists():    
        with open(thresholds_path, "r") as file:
            thresholds = json.load(file)[thresholds_type]
            return thresholds


# initialize dagshub and mlflow
dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)

# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")

# fetch the experiment id
experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

latest_run_id = get_latest_runid(historical_runs_json_path)
latest_metrics = get_run_info(run_id=latest_run_id)

historical_thresholds = load_thresholds(thresholds_type="historical_thresholds",
                                        thresholds_path=threshold_values_path)

noise_thresholds = load_thresholds(thresholds_type="noise_thresholds",
                                   thresholds_path=threshold_values_path)

staging_metrics = get_metrics_from_stage(stage_name=TARGET_STAGE,
                                         experiment_id=experiment_id)


staging_metrics_names = list(staging_metrics.keys())
latest_metrics_names = list(latest_metrics.keys())


def test_similar_metric_names():
    assert staging_metrics_names == latest_metrics_names, "comparison metrics different, use same metrics for comparison only"
    if not staging_metrics_names == latest_metrics_names:
        pytest.exit(reason="Comparison metrics are different")


@pytest.mark.parametrize(argnames="metric",
                         argvalues=latest_metrics_names)
def test_regression_on_metrics(metric: str):
    
    historical_threshold = MULTIPLIER * historical_thresholds[metric]
    noise_threshold = MULTIPLIER * noise_thresholds[metric]
    stage_value = staging_metrics[metric]
    latest_value = latest_metrics[metric]
    
    lower_bound = stage_value - (historical_threshold + noise_threshold)
    assert latest_value > lower_bound, f"Metric {metric} Regressed"
    