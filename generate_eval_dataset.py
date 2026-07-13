from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from rag_workflow import graph
from time import sleep


# load the api keys
load_dotenv()

# create paths
ROOT_DIR = Path(__file__).parent

GOLDENS_PATH = ROOT_DIR / "datasets" / "goldens" / "golden_dataset.json"
EVALUATION_DATA_DIR = GOLDENS_PATH.parent.parent / "eval_dataset"

# create dir
EVALUATION_DATA_DIR.mkdir(exist_ok=True, parents=True)

# evaluation dataset
dataset = EvaluationDataset()
# add goldens to dataset
dataset.add_goldens_from_json_file(
    file_path=GOLDENS_PATH
)

# invoke our application
for golden in dataset.goldens:
    final_state = graph.invoke({"query": golden.input})
    sleep(3)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=final_state.get("response"),
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
    )
    dataset.add_test_case(test_case=test_case)
    
# save the dataset alongg with test cases
dataset.save_as(file_type="json",
                directory=EVALUATION_DATA_DIR,
                file_name="evaluation_dataset",
                include_test_cases=True)
    
    
