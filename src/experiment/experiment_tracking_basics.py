import mlflow
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# set the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# set the experiment name
mlflow.set_experiment("campusx_rag_app")

# start a run
with mlflow.start_run(run_name="test_run") as run:
    # params
    mlflow.log_param("chunk_size", 300)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("output_dims", 1024)
    # metrics
    mlflow.log_metric("recall", 0.89)
    mlflow.log_metric("faithfulness", 0.91)
    mlflow.log_metric("answer_correctness", 0.7)
    # log artifacts
    mlflow.log_artifact(local_path=(ROOT_DIR / "src" / "app" / "rag_workflow.py"),
                        artifact_path="code")
    # log dataset
    mlflow.log_artifact(local_path=(ROOT_DIR / "data" / "evaluation" / "goldens" / "golden_dataset_final.json"))
    
    