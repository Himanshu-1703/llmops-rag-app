from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

# load the api keys
load_dotenv()

rag_input = "What is linear regression?"

expected_output = """Linear regression is a supervised learning algorithm that models the relationship 
between a dependent variable and one or more independent variables by fitting a straight line to 
the data, typically using least squares to minimize the error."""

retrieval_context = [
    "Linear regression assumes a linear relationship between input features and the target variable, and fits a line by minimizing sum of squared residuals.",
    "In linear regression, the model estimates coefficients (slope and intercept) that best fit the training data using the least squares method.",
    "Linear regression is a foundational supervised learning technique used to predict a continuous dependent variable from one or more independent variables.",
    "Logistic regression is used for classification tasks, not regression, despite the name.",
    "Decision trees split data based on feature thresholds to minimize impurity at each node.",
    ]


# test case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    expected_output=expected_output,
    actual_output=""
)

# metric
context_precision= ContextualPrecisionMetric(verbose_mode=True)

# test the metric
context_precision.measure(test_case=test_case)
