from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

# load the api keys
load_dotenv()

rag_input = "How does bagging work?"

expected_output = """Bagging, short for bootstrap aggregating, works by creating 
multiple subsets of the training data through random sampling with 
replacement. A separate model is trained on each subset independently. 
For classification, the final prediction is made by majority voting 
across all models. For regression, the predictions are averaged. This 
process reduces variance and helps prevent overfitting."""

retrieval_context = [
    "Bagging creates multiple bootstrap samples from the training dataset — each sample is drawn randomly with replacement, meaning some data points may appear multiple times while others may be left out.",
    "In bagging, an independent model is trained on each bootstrap sample. For regression tasks, the final output is the average of all individual model predictions.",
    "Random forests are an extension of bagging that also introduces random feature selection at each split, in addition to bootstrap sampling."
    ]

# test case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    expected_output=expected_output,
    actual_output=""
)

# metric
recall = ContextualRecallMetric(verbose_mode=True)

# test the metric
recall.measure(test_case=test_case)