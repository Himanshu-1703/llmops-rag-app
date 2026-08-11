import mlflow
import json
from pathlib import Path
import dagshub


# set paths for the json file
ROOT_DIR = Path(__file__).parent.parent
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"


def log_run_info(run_id: str, run_name: str):
    if JSON_FILE_PATH.exists():
        with open(JSON_FILE_PATH, "r") as file:
            historical_runs = json.load(file)
            
    else:
        historical_runs = []
        
    run_dict = {"run_id": run_id,
                "run_name": run_name}
    
    historical_runs.append(run_dict)
    
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(historical_runs, file,indent=4)
        
        
        
def get_metrics_from_runs(tag_name: str, experiment_id: str): 
    searched_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                       filter_string=f"tags.phase = '{tag_name}'",
                       output_format="list")
    
    all_metrics = []
    
    for run in searched_runs:
        metrics_dict = run.data.metrics
        all_metrics.append(metrics_dict)
        
    return all_metrics


def get_metrics_from_stage(stage_name: str, experiment_id: str): 
    searched_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                       filter_string=f"tags.stage = '{stage_name}'",
                       output_format="list")
    
    metrics = searched_runs[0].data.metrics
    return metrics


def get_run_info(run_id: str):
    run = mlflow.get_run(run_id)
    run_metrics = run.data.metrics
    
    return run_metrics
    
    
if __name__ == "__main__":
    
    # initialize dagshub and mlflow
    dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)
    

    # set the tracking server
    mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")
    
    # fetch the experiment id
    experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id