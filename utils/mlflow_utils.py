import mlflow
import json
from pathlib import Path
import dagshub
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# set paths for the json file
ROOT_DIR = Path(__file__).parent.parent
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"


# champion / challenger promotion tag scheme.
# One tag key `stage`; a run is either the reigning best (champion), the
# candidate under evaluation (challenger), or a dethroned former champion
# (archived). Invariants: exactly one champion, at most one challenger.
STAGE_TAG = "stage"
CHAMPION = "champion"
CHALLENGER = "challenger"
ARCHIVED = "archived"


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
    return get_run_by_stage(stage_name, experiment_id).data.metrics


def get_run_info(run_id: str):
    run = mlflow.get_run(run_id)
    run_metrics = run.data.metrics

    return run_metrics


def find_runs_by_stage(stage_name: str, experiment_id: str) -> list:
    """Every run in the experiment currently carrying `stage=<stage_name>`."""
    return mlflow.search_runs(experiment_ids=[experiment_id],
                              filter_string=f"tags.{STAGE_TAG} = '{stage_name}'",
                              output_format="list")


def get_run_by_stage(stage_name: str, experiment_id: str):
    """The single run tagged `stage=<stage_name>`; raises unless exactly one exists."""
    runs = find_runs_by_stage(stage_name, experiment_id)
    if len(runs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 run tagged {STAGE_TAG}={stage_name!r}, found {len(runs)}. "
            f"Fix the tags in the MLflow UI (bootstrap the champion, or clear stray challengers)."
        )
    return runs[0]


def set_stage(run_id: str, stage_name: str) -> None:
    """Overwrite the `stage` tag on an arbitrary run."""
    MlflowClient().set_tag(run_id, STAGE_TAG, stage_name)


def clear_stage(run_id: str) -> None:
    """Remove the `stage` tag from a run; a missing tag is not an error."""
    try:
        MlflowClient().delete_tag(run_id, STAGE_TAG)
    except MlflowException:
        pass


def demote_stale_challengers(experiment_id: str) -> int:
    """Pre-flight: clear the `stage` tag off any leftover challenger run. Returns count cleared."""
    stale = find_runs_by_stage(CHALLENGER, experiment_id)
    for run in stale:
        clear_stage(run.info.run_id)
    return len(stale)


def promote_challenger(experiment_id: str) -> tuple[str, str]:
    """Swap: reigning champion -> archived, challenger -> champion. Returns (old_id, new_id)."""
    champion = get_run_by_stage(CHAMPION, experiment_id)
    challenger = get_run_by_stage(CHALLENGER, experiment_id)

    set_stage(champion.info.run_id, ARCHIVED)
    set_stage(challenger.info.run_id, CHAMPION)

    remaining = find_runs_by_stage(CHAMPION, experiment_id)
    if len(remaining) != 1:
        raise RuntimeError(
            f"Post-promotion invariant broken: {len(remaining)} runs tagged {STAGE_TAG}={CHAMPION!r}. "
            f"Inspect the MLflow UI immediately."
        )
    return champion.info.run_id, challenger.info.run_id


def reject_challenger(experiment_id: str) -> str:
    """Clear the `stage` tag off the current challenger; champion untouched. Returns the run id."""
    challenger = get_run_by_stage(CHALLENGER, experiment_id)
    clear_stage(challenger.info.run_id)
    return challenger.info.run_id



if __name__ == "__main__":
    
    # initialize dagshub and mlflow
    dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)
    

    # set the tracking server
    mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")
    
    # fetch the experiment id
    experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id