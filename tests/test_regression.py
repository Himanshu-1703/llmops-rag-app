import pytest
from pathlib import Path
import json
import mlflow
import dagshub
from utils.mlflow_utils import get_metrics_from_stage, get_run_by_stage, CHAMPION, CHALLENGER
from typing import Literal

ROOT_DIR = Path(__file__).parent.parent
threshold_values_path = ROOT_DIR / "thresholds.json"

MULTIPLIER = 2


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

latest_metrics = get_run_by_stage(CHALLENGER, experiment_id).data.metrics

historical_thresholds = load_thresholds(thresholds_type="historical_thresholds",
                                        thresholds_path=threshold_values_path)

noise_thresholds = load_thresholds(thresholds_type="noise_thresholds",
                                   thresholds_path=threshold_values_path)

champion_metrics = get_metrics_from_stage(stage_name=CHAMPION,
                                          experiment_id=experiment_id)


champion_metrics_names = list(champion_metrics.keys())
latest_metrics_names = list(latest_metrics.keys())


def test_similar_metric_names():
    assert champion_metrics_names == latest_metrics_names, "comparison metrics different, use same metrics for comparison only"
    if not champion_metrics_names == latest_metrics_names:
        pytest.exit(reason="Comparison metrics are different")


@pytest.mark.parametrize(argnames="metric",
                         argvalues=latest_metrics_names)
def test_regression_on_metrics(metric: str):

    historical_threshold = MULTIPLIER * historical_thresholds[metric]
    noise_threshold = MULTIPLIER * noise_thresholds[metric]
    stage_value = champion_metrics[metric]
    latest_value = latest_metrics[metric]
    
    lower_bound = stage_value - (historical_threshold + noise_threshold)
    assert latest_value > lower_bound, f"Metric {metric} Regressed"
    