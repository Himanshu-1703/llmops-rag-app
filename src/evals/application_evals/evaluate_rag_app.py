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
    name="answer correctness",  # unchanged — feeds MLflow / thresholds.json keys
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    evaluation_steps=[
        "Identify the key facts in 'Expected Output' that answer 'Input': its main claim, "
        "plus any names, numbers, or enumerated items it states.",
        "For each key fact, check 'Actual Output': it conveys the fact, contradicts it "
        "(opposite claim, wrong name or number, reversed cause and effect), or does not "
        "mention it. A key fact that is merely not mentioned is an omission, not an error.",
        "For claims in 'Actual Output' that go beyond 'Expected Output', check them against "
        "'Retrieval Context'. Count such a claim against the response only if it "
        "contradicts 'Retrieval Context' or 'Expected Output', or is clearly false. Extra "
        "detail that is supported by 'Retrieval Context', or is plausible and "
        "uncontradicted, is NOT an error and must not lower the score.",
        "Do not penalise differences in wording, ordering, formatting, length, or the "
        "presence of extra correct information.",
        "Rank issues by severity: contradicting a key fact on a topic 'Input' directly "
        "asks about is the most serious; failing to give a key fact that 'Input' asks for "
        "(including replying 'unknown' or 'not covered' when 'Expected Output' gives a "
        "definite answer) is next; omitting or slightly misstating secondary detail is "
        "minor; extra grounded detail is not an issue.",
        "Pick the rubric band for the most severe issue found; within that band, score "
        "toward the top when the rest of the answer is accurate and complete.",
    ],
    rubric=[
        Rubric(score_range=(0, 3), expected_outcome="The main claim is absent or contradicted, OR 'Actual Output' states something that contradicts a key fact of 'Expected Output' on a topic 'Input' directly asks about, OR several key facts are wrong."),
        Rubric(score_range=(4, 6), expected_outcome="Everything 'Actual Output' states about the key points is correct, but it omits a key fact that 'Input' asks for (including answering 'unknown'/'not covered' when 'Expected Output' is definite), OR one secondary fact is imprecise, OR an added claim contradicts 'Retrieval Context'."),
        Rubric(score_range=(7, 8), expected_outcome="Every key fact is conveyed correctly and nothing is contradicted; at most minor secondary detail is omitted or slightly imprecise. Extra detail supported by 'Retrieval Context' or plausible and uncontradicted is fine."),
        Rubric(score_range=(9, 10), expected_outcome="Every key fact from 'Expected Output' is conveyed correctly with no contradictions and no false statements; wording, length, and extra grounded detail may differ freely."),
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