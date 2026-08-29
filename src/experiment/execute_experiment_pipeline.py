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
import dagshub
import json
from utils.mlflow_utils import log_run_info, demote_stale_challengers

# load the api keys
load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent


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
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    
    if not dir_path.is_dir():
        raise ValueError("Provided path is not a directory")

    filenames = []
    
    files = dir_path.glob(pattern)
    for file in files:
        filenames.append(file.stem)
        
    return max(filenames)


def get_metrics_from_results(results_json_path: Path | str) -> dict:
    metrics = {}
    
    with open(results_json_path, "r") as file:
        metrics_results = json.load(file)["metricsScores"]
        
    for result in metrics_results:
        metric_name = result["metric"].removesuffix("[GEval]") if "[GEval]" in result["metric"] else result["metric"].strip()
        scores = result["scores"]
        avg_scores = round((sum(scores) / len(scores)), 2)
        
        metrics[metric_name] = avg_scores
    
    return metrics



def import_system_prompt(label: str="staging") -> str:
    langfuse = get_client()
    sys_prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=label
    )
    
    return sys_prompt.prompt



def get_artifact_name(artifact_type: Literal["eval_dataset", "golden_dataset"], save_artifact_dir: str) -> tuple[str, str]:
    DATA_PATH = ROOT_DIR / "data" / "evaluation"
    
    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH / "eval_dataset" /  params_config.evaluation_dataset.evaluation_dataset_filename).with_suffix(".json")
        return (artifact_path, save_artifact_dir)
    
    elif artifact_type == "golden_dataset":
        artifact_path = (DATA_PATH / "goldens" /  params_config.golden_dataset.golden_dataset_filename).with_suffix(".json")
        return (artifact_path, save_artifact_dir)
    
    
def return_code_files():
    CODE_PATHS = ROOT_DIR / "src"
    
    app_code_path = CODE_PATHS / "app" /"rag_workflow.py"
    clients_code_path = CODE_PATHS / "app" / "clients.py"
    vector_store_code_path = CODE_PATHS / "app" / "vector_store.py"
    eval_data_path = CODE_PATHS / "data" / "generate_eval_dataset.py"
    evaluation_path = CODE_PATHS / "evals" / "application_evals" / "evaluate_rag_app.py"
    golden_dataset_path = CODE_PATHS / "data" / "generate_goldens.py"

    code_files = [app_code_path, clients_code_path, vector_store_code_path, eval_data_path, evaluation_path, golden_dataset_path]
    
    return [file.as_posix() for file in code_files]


if __name__ == "__main__":
    # initialize dagshub and mlflow
    dagshub.init(repo_owner='himanshu1703', repo_name='llmops-rag-app', mlflow=True)
    
    
    # set the tracking server
    mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/llmops-rag-app.mlflow")
    
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

    # pre-flight: this run becomes the new challenger, so clear the stage tag
    # off any leftover challenger from a prior (failed or superseded) cycle.
    experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id
    demoted = demote_stale_challengers(experiment_id)
    logger.info(f"Pre-flight: demoted {demoted} stale challenger run(s)")

    with mlflow.start_run() as run:
    
        # get all the params
        all_params = params_config.model_dump()
        params_dict = flatten_params(all_params)
        
        # log params on mlflow
        mlflow.log_params(params_dict)
        logger.info("Parameters Logged")
        
        # generate evaluation data
        generate_evaluation_dataset()
        logger.info("evaluation dataset created")
        
        #run the eval pipeline
        evaluate_app()
        logger.info("evaluation complete")
        
        # paths for latest reports
        RESULT_DIR = ROOT_DIR / "reports" / "evaluation_results"
        REPORT_DIR = ROOT_DIR / "reports" / "evaluation_report"
        
        # get the latest files after eval pipeline
        latest_results_filename = get_latest_results(RESULT_DIR, "*.json")
        latest_report_filename = get_latest_results(REPORT_DIR, "*.md")

        results_file = (RESULT_DIR / latest_results_filename).with_suffix(".json").as_posix()
        report_file = (REPORT_DIR / latest_report_filename).with_suffix(".md").as_posix()
        
        # log the evaluation results
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
                                            save_artifact_dir="eval_dataset")
        golden_dataset_artifact = get_artifact_name(artifact_type="golden_dataset",
                                                save_artifact_dir="golden_dataset")
        mlflow.log_artifact(eval_dataset_artifact[0], eval_dataset_artifact[1])
        mlflow.log_artifact(golden_dataset_artifact[0], golden_dataset_artifact[1])
        logger.info("logged evaluation and golden datasets")
        
        # log the code files
        code_files = return_code_files()
        
        for code_file in code_files:
            mlflow.log_artifact(code_file, "code")
        logger.info("code files logged")
    
        # set tag for the run
        mlflow.set_tag("phase", "historical_threshold")

        # mark this run as the candidate for the regression / promotion gates
        mlflow.set_tag("stage", "challenger")
        logger.info("Run tagged stage=challenger")
        
    # extract info from run
    run_id = run.info.run_id
    run_name = run.info.run_name
    
    # log to json file
    log_run_info(run_id, run_name)