import numpy as np
from pydantic import BaseModel, Field

from guardrails import AsyncGuard
from guardrails.classes import ValidationOutcome
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)
from guardrails_ai.detect_jailbreak import DetectJailbreak
from guardrails_ai.gibberish_text import GibberishText
from guardrails_ai.guardrails_pii import GuardrailsPII
from guardrails_ai.prompt_injection_detector import PromptInjectionDetector
from guardrails_ai.reading_time import ReadingTime
from guardrails_ai.redundant_sentences import RedundantSentences
from guardrails_ai.relevancy_evaluator import RelevancyEvaluator
from guardrails_ai.provenance_embeddings import ProvenanceEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate



class TopicResult(BaseModel):
    score: float = Field(
        description="Relevance score from 0.0 (off-topic) to 1.0 (strongly on-topic)"
    )
    topic: str = Field(
        description="The exact matched topic from the valid list, or 'None' if off-topic"
    )


@register_validator(name="on_topic", data_type="string")
class OnTopic(Validator):
    def __init__(
        self,
        valid_topics: list,
        threshold: float = 0.7,
        on_fail: str | None = None,
    ):
        super().__init__(
            on_fail=on_fail,
            valid_topics=valid_topics,
            threshold=threshold,
        )
        self.valid_topics = valid_topics
        self.threshold = threshold

        # Initialize LLM chain
        self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0).with_structured_output(
            schema=TopicResult
        )
        self.prompt = PromptTemplate.from_template(
            """You are a topic classification validator.

            Evaluate whether the user query relates to any of the valid topics:
            - Valid Topics: {valid_topics}
            - User Query: "{query}"

            Instructions:
            1. Assign a relevance score `score` between 0.0 and 1.0:
            - 0.8 to 1.0: The query directly matches one of the valid topics.
            - 0.0 to 0.4: The query is off-topic, unrelated, or conversational chit-chat.
            2. Set `topic` to the exact matching topic from the list, or 'None' if off-topic."""
        )
        self.chain = self.prompt | self.llm

    def _validate(self, value: str, metadata: dict) -> ValidationResult:

        response: TopicResult = self.chain.invoke(
            {
                "query": value,
                "valid_topics": ", ".join(self.valid_topics),
            }
        )

        if response.score >= self.threshold and response.topic in self.valid_topics:
            return PassResult()

        return FailResult(
            error_message=(
                f"Query '{value}' is off-topic. "
                f"Best match: '{response.topic}' with score {response.score:.2f} "
                f"(threshold: {self.threshold})."
            )
        )


# --- Layer 1: input guardrails (user query, before retrieval) 

PII_ENTITIES = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "CREDIT_CARD"]

VALID_TOPICS = [
    "Evals", "Evaluation", "LLM", "AI", "Generative AI", "LLMOps",
    "AI News", "Tech", "AI Engineer", "Code", "Python",
]

input_guard = AsyncGuard().use(
    DetectJailbreak(on_fail="exception"),
    GibberishText(threshold=0.5, validation_method="sentence", on_fail="refrain"),
    GuardrailsPII(entities=PII_ENTITIES, on_fail="fix"),
    OnTopic(threshold=0.7, valid_topics=VALID_TOPICS, on_fail="exception"),
)


async def validate_input(query: str) -> ValidationOutcome:
    return await input_guard.validate(query)


# --- Layer 2: retrieval guardrails (retrieved context, before augmentation)

retrieval_guard = AsyncGuard().use(
    PromptInjectionDetector(llm_callable="gpt-5-mini", on_fail="exception"),
    RedundantSentences(on_fail="refrain"),
)


async def validate_retrieval(context: str) -> ValidationOutcome:
    return await retrieval_guard.validate(context)


# --- Layer 3: output guardrails (generated response, before it's returned)

def embed_func(texts):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if isinstance(texts, str):
        return np.array(embedding_model.embed_query(texts))

    return np.array(embedding_model.embed_documents(texts))


output_guard = AsyncGuard().use(
    ReadingTime(reading_time=3, on_fail="noop"),
    RelevancyEvaluator(llm_callable="gpt-5-mini", on_fail="refrain"),
    ProvenanceEmbeddings(threshold=0.7, validation_method="sentence", on_fail="refrain"),
)


async def validate_output(response: str, *, sources: list[str], query: str) -> ValidationOutcome:
    
    metadata = {
        "sources": sources,
        "original_prompt": query,
        "embed_function": embed_func,
        "chunk_size": 2,
        "chunk_overlap": 1,
    }
    return await output_guard.validate(response, metadata=metadata)
