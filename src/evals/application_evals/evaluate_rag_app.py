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
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from pathlib import Path
from dotenv import load_dotenv
from config.parameter_config import params_config

# load the evaluation params
evaluation_params = params_config.evaluation
async_params = evaluation_params.async_config
display_params = evaluation_params.display_config
evaluation_dataset_params = params_config.evaluation_dataset

# load the api keys
load_dotenv()

model = evaluation_params.judge_llm

# define the metrics
recall = ContextualRecallMetric(model=model)
precision = ContextualPrecisionMetric(model=model)
contextual_relevancy = ContextualRelevancyMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model)
faithfulness = FaithfulnessMetric(model=model)

# define the custom metrics
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "List every factual claim in 'Expected Output': definitions, names, numbers, "
        "cause-and-effect statements, and each item in any enumeration.",
        "For each claim, decide whether 'Actual Output' states the same fact, contradicts "
        "it, or omits it. A wrong name, wrong number, or reversed cause and effect counts "
        "as a contradiction.",
        "Check whether 'Actual Output' adds claims that are not in 'Expected Output' and "
        "not asked for by 'Input'; mark any that are factually wrong or misleading as "
        "fabrications.",
        "Do not penalise differences in wording, ordering, formatting, length, or extra "
        "detail that is correct and relevant to 'Input'.",
        "Weight each issue by how central it is to answering 'Input': a contradicted or "
        "missing core fact matters more than a peripheral one.",
        "Select the rubric band matching the most severe issue found.",
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="A core fact needed to answer 'Input' is contradicted, or the main point of 'Expected Output' is wrong or absent."),
        Rubric(score_range=(3, 5), expected_outcome="The main point is broadly right, but at least one supporting fact is contradicted or a misleading fabrication is present."),
        Rubric(score_range=(6, 8), expected_outcome="All stated facts are correct and the main point matches; one or more secondary facts from 'Expected Output' are omitted or slightly imprecise."),
        Rubric(score_range=(9, 10), expected_outcome="Every fact in 'Expected Output' is present and correct, with no contradictions and no fabrications; wording and extra correct detail may differ."),
    ],
    model=model
)

simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read 'Input' to see what was asked, then read 'Actual Output' as a student new to "
        "the topic.",
        "Find the technical terms and jargon in 'Actual Output'. For each, check whether it "
        "is avoided, replaced with plain language, or explained in-line the first time it "
        "is used.",
        "Judge structural complexity: short sentences, concrete examples, analogies, and "
        "lists make it more accessible; long nested sentences and dense abstraction make it "
        "less accessible.",
        "Decide whether a beginner could follow the explanation from start to finish "
        "without outside knowledge.",
        "Judge only readability and accessibility, not factual accuracy (that is scored "
        "separately). The one exception: do not reward simplicity achieved by omitting "
        "parts of what 'Input' asked for.",
        "Select the rubric band matching how much effort a beginner needs to understand "
        "'Actual Output'.",
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Dense with unexplained jargon or convoluted sentences; a beginner cannot follow it."),
        Rubric(score_range=(3, 5), expected_outcome="Several unexplained technical terms or tangled sentences; a beginner understands only parts."),
        Rubric(score_range=(6, 8), expected_outcome="Mostly plain language with only a few unexplained terms; a beginner can follow it with some effort."),
        Rubric(score_range=(9, 10), expected_outcome="Plain language throughout, jargon avoided or explained, aided by examples or analogies; a beginner follows it easily."),
    ],
    model=model
)

def evaluate_app():

    # define the dataset path
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_PATH = (ROOT_DIR / "data" / "evaluation" / "eval_dataset" / evaluation_dataset_params.evaluation_dataset_filename).with_suffix(".json")

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
                async_config=AsyncConfig(throttle_value=async_params.throttle_value,
                                        max_concurrent=async_params.max_concurrent),
                display_config=DisplayConfig(results_folder=(ROOT_DIR / "reports" / display_params.results_dir).as_posix(),
                                            file_type="md",
                                            file_output_dir=(ROOT_DIR / "reports" / display_params.report_dir).as_posix())
        )


if __name__ == "__main__":
    evaluate_app()