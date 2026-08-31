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
        "Identify the PRIMARY thing 'Input' asks for (its central question), separately "
        "from any secondary elaboration it invites.",
        "Build the list of 'key facts': claims in 'Expected Output' that answer 'Input' AND "
        "for which you can point to a specific supporting sentence in 'Retrieval Context'. "
        "If you cannot locate a supporting sentence in 'Retrieval Context' for a claim in "
        "'Expected Output', it is NOT a key fact: the system could not have known it, so "
        "its absence from 'Actual Output' — including an explicit 'I don't know' or 'the "
        "context does not say' about it — is never an omission or an error and must not "
        "lower the score.",
        "'Retrieval Context' outranks 'Expected Output' as the standard for the primary "
        "answer. If 'Expected Output' frames the central cause / mechanism / answer in a "
        "way that 'Retrieval Context' does not actually support, judge 'Actual Output' "
        "against the retrieval-supported framing instead. An 'Actual Output' statement "
        "that is faithful to 'Retrieval Context' but conflicts only with an unsupported "
        "generalisation in 'Expected Output' is at most a minor secondary imprecision "
        "(7-8 band), never a contradiction.",
        "For each key fact, check whether 'Actual Output' conveys it, contradicts it "
        "(opposite claim, wrong name or number, reversed cause and effect), or omits it.",
        "Judge contradictions against BOTH 'Expected Output' and 'Retrieval Context'. If "
        "'Actual Output' diverges from a nuance in 'Expected Output' but stays consistent "
        "with 'Retrieval Context', that is NOT a contradiction and must not lower the "
        "score.",
        "Extra claims in 'Actual Output' that go beyond 'Expected Output' count against it "
        "ONLY if they contradict 'Retrieval Context' or 'Expected Output', or are clearly "
        "false. Extra detail that is grounded in 'Retrieval Context', OR merely plausible "
        "and uncontradicted, is NOT an error: it must not lower the score and must not cap "
        "it below the top band, no matter how much of it there is or how loosely grounded "
        "it is.",
        "Do not penalise differences in wording, ordering, formatting, length, or "
        "structure (prose vs JSON vs bullets).",
        "Rank issues by severity: contradicting a key fact is most serious; failing to "
        "convey a key fact that answers the PRIMARY question is next; omitting or slightly "
        "misstating a secondary retrieval-supported detail is minor and belongs in the "
        "7-8 band, not lower; extra detail and anything not supported by 'Retrieval "
        "Context' are not issues at all.",
        "Pick the rubric band for the most severe real issue found; within that band, "
        "score at the top when the rest of the answer is accurate.",
    ],
    rubric=[
        Rubric(score_range=(0, 3), expected_outcome="The main claim is absent or contradicted, OR 'Actual Output' contradicts a key fact (one you can point to in 'Retrieval Context') on a topic 'Input' directly asks about, OR several key facts are wrong."),
        Rubric(score_range=(4, 6), expected_outcome="Every key fact 'Actual Output' addresses is correct, but it fails to convey a key fact that answers the PRIMARY question of 'Input' and that 'Retrieval Context' supports, OR an added claim directly contradicts 'Retrieval Context'."),
        Rubric(score_range=(7, 8), expected_outcome="Every key fact answering the primary question is conveyed correctly and nothing contradicts 'Retrieval Context'; at most one secondary retrieval-supported detail is missing or slightly imprecise, OR one statement is faithful to 'Retrieval Context' but conflicts with an unsupported generalisation in 'Expected Output'."),
        Rubric(score_range=(9, 10), expected_outcome="Every key fact that 'Input' primarily asks for and that 'Retrieval Context' supports is conveyed correctly, with no contradiction of 'Expected Output' or 'Retrieval Context' and no clearly false statement. Wording, length, format, and any amount of extra grounded or plausible-uncontradicted detail may differ freely. A claim in 'Expected Output' with no locatable support in 'Retrieval Context' is not required; its absence, or an explicit 'not covered' about it, does not lower the score. Loosely-grounded-but-uncontradicted additions also stay in this band."),
    ],
    model=model
)

simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read 'Input' to see what was asked and which terms the asker already uses. Read "
        "'Actual Output' as an LLMOps student: someone taking a course on building and "
        "operating LLM applications, comfortable with everyday AI/software vocabulary but "
        "new to fine detail.",
        "Find the OPAQUE JARGON in 'Actual Output': terms an LLMOps student would still "
        "not understand. The following do NOT count as opaque jargon and must not lower "
        "the score: (a) any term already used in 'Input'; (b) ordinary English words and "
        "descriptive phrases even if slightly domain-flavoured (e.g. 'content team', "
        "'recorded courses', 'complementary', 'remediation', 'auto-fixing', 'cyber "
        "partners'); (c) standard AI/RAG-pipeline vocabulary this audience already knows "
        "(e.g. 'model', 'prompt', 'system prompt', 'probabilistic', 'deterministic', "
        "'retrieval', 'retriever', 'RAG', 'embedding'/'embedder', 're-ranker', 'vector "
        "store', 'chunking', 'NLU', 'token', 'latency', 'training data'); (d) proper "
        "nouns / product names, which need no definition.",
        "For each piece of opaque jargon, check whether it is avoided, replaced with plain "
        "language, or made clear in-line or from surrounding context (an example, an "
        "appositive, a parenthetical) the first time it is used. An acronym or term whose "
        "meaning is clear from context counts as handled.",
        "Judge structure: short sentences, headings, bullet lists, concrete examples, and "
        "clear step-by-step breakdowns make it accessible; long nested sentences and "
        "unbroken dense paragraphs make it less accessible. An analogy is a bonus, never a "
        "requirement.",
        "Judge only readability and accessibility, not factual accuracy (that is scored "
        "separately). The one exception: do not reward simplicity achieved by omitting "
        "parts of what 'Input' asked for.",
        "Decide how much effort an LLMOps student needs to follow 'Actual Output' from "
        "start to finish, and pick the matching band.",
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Dense with unhandled opaque jargon or convoluted sentences; the student cannot follow it."),
        Rubric(score_range=(3, 5), expected_outcome="Three or more pieces of opaque jargon are left unhandled, or the structure is tangled; the student follows only parts."),
        Rubric(score_range=(6, 8), expected_outcome="At least one opaque term that is essential to the main point is left unhandled AND this genuinely impedes understanding, or the structure is dense and hard to scan; the student needs real extra effort."),
        Rubric(score_range=(9, 10), expected_outcome="The answer is well organised (short sentences, headings/bullets, at least one concrete example or clear breakdown where it helps) and any unhandled terms are either non-opaque to this audience or inessential to the main point. An LLMOps student can follow it from start to finish. Reusing the Input's own terminology, ordinary domain-flavoured phrasing, standard AI/RAG vocabulary, and product names is expected and fine."),
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