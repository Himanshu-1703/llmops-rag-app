from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from dotenv import load_dotenv

# load the api key
load_dotenv()

# define the custom metric
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare
    the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0,5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
    verbose_mode=True
)

test_case = LLMTestCase(
    input="",
    expected_output="The 3 question-paper setup approaches are:\n\n1. **Manual creation**\n   - **Pros:** Human control, can ensure coverage and balanced difficulty.\n   - **Cons:** Biased, costly, hard to scale, depends on human availability, and breaks when the knowledge base updates.\n\n2. **Historical data**\n   - **Pros:** Uses real previous user questions, so it reflects actual usage.\n   - **Cons:** May have low coverage, still needs correct answers for scoring, and can be limited by one type of user behavior.\n\n3. **Synthetic data**\n   - **Pros:** Automated, scalable, lower cost, less bias, no fatigue/illness issues, and adapts better to knowledge base updates.\n   - **Cons:** Needs a smart generation method, which was only briefly mentioned in the context.",
    actual_output="- Manual (human-crafted question papers)\n  - Pro: can create tailored exams across multiple subjects.\n  - Con: high manual effort and poor scalability (you must create many exams/subjects by hand).\n\n- Synthetic data (automatically generated Q&A pairs)\n  - Pro: generate question–answer pairs programmatically (e.g., using a framework like DeepEval) to set the question paper — reduces manual creation.\n  - Con: no specific downsides are mentioned in the provided context.\n\n- Historical data (reuse previous-year question papers)\n  - Pro: reuse/fetch existing questions from past papers so you can assemble papers without creating everything anew.\n  - Con: no specific downsides are mentioned in the provided context.\n\n(Additional context notes mentioned: evaluation is based on retriever accuracy, tests may run offline or in production, and tools like RAGas are part of the LLM eval setup.)",
)

print(answer_correctness.measure(test_case=test_case))