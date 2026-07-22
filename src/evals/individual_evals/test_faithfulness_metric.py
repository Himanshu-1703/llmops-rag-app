from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

# load the api keys
load_dotenv()

rag_input = "Explain how the k-Nearest Neighbors algorithm works, including how the value of k affects performance."

actual_output = """k-Nearest Neighbors is a non-parametric algorithm that can be used for classification and regression tasks. It predicts the label of a query point by looking at the k closest training points, usually measured using Euclidean distance, and taking a majority vote or average. Since kNN doesn't learn an explicit model during training and just stores the data, it's often referred to as a lazy learner. Choosing a very small value of k typically makes the model more robust to noisy data. Feature scaling matters a lot for kNN since it relies on distance calculations. kNN has a training time complexity of O(n log n) due to the tree structures used internally."""


retrieval_context = [
    "k-Nearest Neighbors is a non-parametric, instance-based learning algorithm used for both classification and regression.",
    "It works by finding the k closest training examples to a query point, typically using Euclidean distance, and predicting based on majority vote (for classification) or averaging (for regression).",
    "The algorithm does not build an explicit model during training — it simply stores the training data, which is why it is called a 'lazy learner.'",
    "The choice of k significantly affects performance: a small k makes the model sensitive to noise, while a large k can smooth out decision boundaries too much.",
    "Feature scaling is important for kNN because distance calculations are sensitive to the magnitude of feature values."
]



# test case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    actual_output=actual_output
)

# metric
faithfulness = FaithfulnessMetric(verbose_mode=True,
                                  penalize_ambiguous_claims=False,
                                  truths_extraction_limit=7)

# test the metric
faithfulness.measure(test_case=test_case)