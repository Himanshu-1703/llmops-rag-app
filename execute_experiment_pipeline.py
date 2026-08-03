from data.generate_eval_dataset import generate_evaluation_dataset
from evals.application_evals.evaluate_rag_app import evaluate_app
from pathlib import Path
from config.parameter_config import params_config
from dotenv import load_dotenv
from langfuse import get_client
from typing import Literal
import json
import mlflow
import logging
from logging import INFO

# laod the api keys
load_dotenv()

ROOT_DIR = Path(__file__).parent


def flatten_params(params_dict: dict) -> dict:
    config_dict = {}
    
    for key, value in params_dict.items():
        if isinstance(value, dict):
            config_dict.update(flatten_params(value))
        else:
            if key in config_dict:
                raise ValueError(f"Duplicate key found: {key}")
            config_dict[key] = value
    
    return config_dict


def get_latest_results(dir_path: Path | str, pattern: str):
    if not dir_path.is_dir():
        raise ValueError("Provided path is not a directory")

    filenames = []
    
    files = dir_path.glob(pattern)
    for file in files:
        filenames.append(file.stem)
        
    return max(filenames)


def get_metrics_from_results(results_json) -> dict:
    metrics = {}
    
    with open(results_json, "r") as file:
        metrics_results = json.load(file)["metricsScores"]
        
    for result in metrics_results:
        metric_name = result["metric"].removesuffix("[GEval]") if "[GEval]" in result["metric"] else result["metric"].strip()
        scores = result["scores"]
        avg_scores = round((sum(scores) / len(scores)), 2)
        
        metrics[metric_name] = avg_scores
    
    return metrics



def import_system_prompt(label="staging") -> str:
    langfuse = get_client()
    prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=label
    )
    
    return prompt.prompt



def get_artifact_name(artifact_type: Literal["eval_dataset", "golden_dataset"], artifact_dir: str) -> tuple[str, str]:
    DATA_PATH = ROOT_DIR / "data" / "evaluation"
    
    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH / "eval_dataset" /  params_config.evaluation_dataset.evaluation_dataset_filename).with_suffix(".json")
        return (artifact_path, artifact_dir)
    
    elif artifact_type == "golden_dataset":
        artifact_path = (DATA_PATH / "goldens" /  params_config.golden_dataset.golden_dataset_filename).with_suffix(".json")
        return (artifact_path, artifact_dir)
    
    
def return_code_files():
    code_file_paths = ROOT_DIR / "src"
    
    app_code_path = code_file_paths / "app" /"rag_workflow.py"
    eval_data_path = code_file_paths / "data" / "generate_eval_dataset.py"
    evaluation_path = code_file_paths / "evals" / "application_evals" / "evaluate_rag_app.py"
    golden_dataset_path = code_file_paths / "data" / "generate_goldens.py"
    
    return [file.as_posix() for file in [app_code_path, eval_data_path, evaluation_path, golden_dataset_path]]


if __name__ == "__main__":
    
    # set the tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # set the experiment name
    mlflow.set_experiment("rag-app")
    
    # do the logging
    logger = logging.getLogger(name="MLflow logger")
    # add stream handler
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(INFO)
    # add formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt=formatter)
    
    # get all the params
    all_params = params_config.model_dump()
    params_dict = flatten_params(all_params)
    
    mlflow.log_params(params_dict)
    logger.info("Parameters Logged")
    
    # generate evaluation data
    generate_evaluation_dataset()
    logger.info("evaluation dataset created")
    
    #run the eval pipeline
    evaluate_app()
    logger.info("evaluation complete")
    
    # paths for latest reports
    result_dir = ROOT_DIR / "reports" / "evaluation_results"
    report_dir = ROOT_DIR / "reports" / "evaluation_report"
    
    # get the latest files after eval pipeline
    results_file = (result_dir / get_latest_results(result_dir, "*.json")).with_suffix(".json").as_posix()
    report_file = (report_dir / get_latest_results(report_dir, "*.md")).with_suffix(".md").as_posix()
    
    # log the results
    mlflow.log_artifact(results_file, "results")
    mlflow.log_artifact(report_file, "reports")
    logger.info("Results and Report logged")
    
    # log the metrics
    metrics = get_metrics_from_results(results_file)
    mlflow.log_metrics(metrics)
    logger.info("Metrics Logged")
    
    # log the system prompt
    system_prompt = import_system_prompt(params_config.rag_app.prompt_label)
    mlflow.log_text(system_prompt,
                    artifact_file="system_prompt.txt")
    logger.info("System prompt logged")
    
    
    # log the datasets
    eval_dataset_artifact = get_artifact_name(artifact_type="eval_dataset",
                                          artifact_dir="eval_dataset")
    golden_dataset_artifact = get_artifact_name(artifact_type="golden_dataset",
                                            artifact_dir="golden_dataset")
    mlflow.log_artifact(eval_dataset_artifact[0], eval_dataset_artifact[1])
    mlflow.log_artifact(golden_dataset_artifact[0], golden_dataset_artifact[1])
    logger.info("logged evaluation and golden datasets")
    
    # log the code files
    code_files = return_code_files()
    
    for code_file in code_files:
        mlflow.log_artifact(code_file, "code")
    logger.info("code files logged")
    