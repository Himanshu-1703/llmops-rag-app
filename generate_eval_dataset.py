from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from rag_workflow import graph


# load the api keys
load_dotenv()

# create paths
ROOT_DIR = Path(__file__).parent

GOLDENS_PATH = ROOT_DIR / "datasets" / "goldens" / "golden_dataset.json"
EVALUATION_DATA_DIR = GOLDENS_PATH.parent.parent / "eval_dataset"

# create dir
EVALUATION_DATA_DIR.mkdir(exist_ok=True, parents=True)

# dataset to read goldens from
golden_dataset = EvaluationDataset()
golden_dataset.add_goldens_from_json_file(file_path=GOLDENS_PATH)

# dataset to hold produced test cases
eval_dataset = EvaluationDataset()

for golden in golden_dataset.goldens:
    final_state = graph.invoke({"query": golden.input})
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=final_state.get("response"),
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
    )
    eval_dataset.add_test_case(test_case=test_case)

eval_dataset.save_as(
    file_type="json",
    directory=EVALUATION_DATA_DIR,
    file_name="evaluation_dataset",
    include_test_cases=True
)
