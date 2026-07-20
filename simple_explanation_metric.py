from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from dotenv import load_dotenv

# load the api key
load_dotenv()

# define the custom criteria
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
    verbose_mode=True
)

test_case = LLMTestCase(
    input="What is Linear Regression",
    actual_output="Linear regression is a foundational supervised machine learning algorithm used to predict continuous numerical values. It models the linear relationship between a dependent (target) variable and one or more independent (input) variables by fitting the best possible straight line to the data"
)

print(simple_explanation.measure(test_case))