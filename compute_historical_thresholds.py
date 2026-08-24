import dagshub
import mlflow
from utils.mlflow_utils import get_metrics_from_runs
import numpy as np
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent
THRESH_PATH = ROOT_DIR / "thresholds.json"

def save_thresholds(json_path, stds, noise_thresholds = None):
    # read thresholds
    if json_path.exists():
        with open(json_path, "r") as file:
            thresholds = json.load(file)
            noise_thresholds = thresholds.get("noise_thresholds")
            historical_thresholds = thresholds.get("historical_thresholds")
            
    # new thresholds
    new_thresholds = {}
    new_thresholds["noise_thresholds"] = noise_thresholds
    new_thresholds["historical_thresholds"] = stds
    
    # save thresholds    
    with open(json_path, "w") as file:
        json.dump(new_thresholds, file, indent=4)
        


def calculate_historical_threshold(metrics: list[dict[str, float]]):
    all_metrics = []
    
    for metric in metrics:
        metric_names = list(metric.keys())
        metric_values = list(metric.values())
        all_metrics.append(metric_values)
    
    
    matrix = np.array(all_metrics)
    stds = np.std(matrix, axis=0, ddof=1).tolist()
    
    return {key: val for key, val in zip(metric_names, stds)}

# initialize dagshub and mlflow
dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)
    
# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")

experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

# get the metrics across noise thresh runs
all_metrics = get_metrics_from_runs(tag_name="historical_threshold",
                      experiment_id=experiment_id)

stds = calculate_historical_threshold(all_metrics)

print(stds)
save_thresholds(THRESH_PATH, stds)