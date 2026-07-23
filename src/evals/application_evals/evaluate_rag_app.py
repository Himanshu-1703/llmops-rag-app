from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.evaluate import evaluate
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, CacheConfig
from pathlib import Path
from dotenv import load_dotenv

# load the api keys
load_dotenv()

model = "gpt-5.4-mini"

# define the metrics
recall = ContextualRecallMetric(model=model)
precision = ContextualPrecisionMetric(model=model)
contextual_relevancy = ContextualRelevancyMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model)
faithfulness = FaithfulnessMetric(model=model)

# define the custom metrics
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare
    the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0,5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
    model=model
)

simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read the 'actual output' first and then the 'input'",
        "Check whether the response is simple and easy to understand or not",
        "Make sure the response has least number of technical jargons and is student friendly"
    ],
    rubric=[
        Rubric(score_range=(0,3), expected_outcome="Too Difficult"),
        Rubric(score_range=(4,7), expected_outcome="Slightly difficult"),
        Rubric(score_range=(8,9), expected_outcome="moderately simple"),
        Rubric(score_range=(10,10), expected_outcome="very simple")
    ],
    model=model
)

# define the dataset path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "evaluation" / "eval_dataset" / "evaluation_dataset_final.json"

if DATASET_PATH.exists():
    # load the dataset
    dataset = EvaluationDataset()
    
    # load the test cases
    dataset.add_test_cases_from_json_file(
        file_path=DATASET_PATH,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        retrieval_context_key_name="retrieval_context"
    )
    
    # store the test cases in a list
    test_cases = dataset.test_cases
    
    
    # evaluate the dataset
    evaluate(test_cases=test_cases,
             metrics=[recall,
                      precision,
                      answer_relevancy,
                      faithfulness,
                      contextual_relevancy,
                      answer_correctness,
                      simple_explanation],
             async_config=AsyncConfig(throttle_value=3,
                                      max_concurrent=5),
             display_config=DisplayConfig(results_folder=(ROOT_DIR / "reports" / "evaluation_results").as_posix(),
                                          file_type="md",
                                          file_output_dir=(ROOT_DIR / "reports" / "evaluation_report").as_posix()),
    )