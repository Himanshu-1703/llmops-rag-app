import asyncio
from typing import TypedDict, Literal

from guardrails.errors import ValidationError
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client
from langgraph.graph import END, START, StateGraph

from app.clients import app_params, llm, logger
from app.guardrails import validate_input, validate_output, validate_retrieval
from app.vector_store import get_retriever

# langfuse client
langfuse = get_client()

# load system prompt
system_prompt = langfuse.get_prompt(
    name="rag_app_system_prompt",
    type="text",
    label=app_params.prompt_label
)

CRITICAL_FALLBACK_MESSAGE = (
    "Something went wrong while processing your request and it could not be "
    "completed safely. Please try again in a moment."
)
SOFT_FALLBACK_MESSAGE = (
    "I couldn't produce a fully reliable answer to that question. Could you "
    "rephrase it or ask something more specific?"
)


class GuardrailedRAGState(TypedDict):

    query: str
    retrieved_docs: list[Document]
    context: str
    prompt: ChatPromptTemplate
    response: str
    guardrail_status: Literal["ok", "exception", "refrain"]
    guardrail_stage: Literal["input", "retrieval", "output"]
    guardrail_message: str


def retrieve(state: GuardrailedRAGState) -> dict:
    query = state["query"]
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs": retrieved_docs,
            "context": context}


def augmentation(state: GuardrailedRAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.prompt),
        ("human", "context: {context}\n\nquery: {query}")
    ])

    return {"prompt": prompt}


def generation(state: GuardrailedRAGState) -> dict:

    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": context, "query": query})

    return {"response": response}


def input_guardrail_node(state: GuardrailedRAGState) -> dict:
    query = state["query"]
    try:
        outcome = asyncio.run(validate_input(query))
    except ValidationError as e:
        logger.error(f"[input_guardrail] exception: {e}")
        return {
            "guardrail_status": "exception",
            "guardrail_stage": "input",
            "guardrail_message": str(e),
        }

    if not outcome.validation_passed:
        logger.warning(f"[input_guardrail] refrain: {outcome.error}")
        return {
            "guardrail_status": "refrain",
            "guardrail_stage": "input",
            "guardrail_message": outcome.error or "Input failed validation.",
        }

    # GuardrailsPII uses on_fail="fix" -- validated_output carries the
    # PII-redacted query. Forward it so every downstream node retrieves
    # and generates against the redacted text, not the raw original.
    return {
        "query": outcome.validated_output or query,
        "guardrail_status": "ok",
        "guardrail_stage": "input",
        "guardrail_message": "",
    }


def retrieval_guardrail_node(state: GuardrailedRAGState) -> dict:
    context = state["context"]
    try:
        outcome = asyncio.run(validate_retrieval(context))
    except ValidationError as e:
        logger.error(f"[retrieval_guardrail] exception: {e}")
        return {
            "guardrail_status": "exception",
            "guardrail_stage": "retrieval",
            "guardrail_message": str(e),
        }

    if not outcome.validation_passed:
        logger.warning(f"[retrieval_guardrail] refrain: {outcome.error}")
        return {
            "guardrail_status": "refrain",
            "guardrail_stage": "retrieval",
            "guardrail_message": outcome.error or "Retrieved context failed validation.",
        }

    return {
        "guardrail_status": "ok",
        "guardrail_stage": "retrieval",
        "guardrail_message": "",
    }


def output_guardrail_node(state: GuardrailedRAGState) -> dict:
    response = state["response"]
    query = state["query"]
    sources = [doc.page_content for doc in state["retrieved_docs"]]
    try:
        outcome = asyncio.run(validate_output(response, sources=sources, query=query))
    except ValidationError as e:
        logger.error(f"[output_guardrail] exception: {e}")
        return {
            "guardrail_status": "exception",
            "guardrail_stage": "output",
            "guardrail_message": str(e),
        }

    if not outcome.validation_passed:
        logger.warning(f"[output_guardrail] refrain: {outcome.error}")
        return {
            "guardrail_status": "refrain",
            "guardrail_stage": "output",
            "guardrail_message": outcome.error or "Response failed validation.",
        }

    return {
        "guardrail_status": "ok",
        "guardrail_stage": "output",
        "guardrail_message": "",
    }


def guardrail_router(state: GuardrailedRAGState) -> str:
    return state["guardrail_status"]


def error_fallback_node(state: GuardrailedRAGState) -> dict:
    logger.error(
        f"[error_fallback] stage={state.get('guardrail_stage')} "
        f"message={state.get('guardrail_message')}"
    )
    return {
        "response": CRITICAL_FALLBACK_MESSAGE,
        "retrieved_docs": state.get("retrieved_docs", []),
    }


def soft_fallback_node(state: GuardrailedRAGState) -> dict:
    logger.warning(
        f"[soft_fallback] stage={state.get('guardrail_stage')} "
        f"message={state.get('guardrail_message')}"
    )
    return {
        "response": SOFT_FALLBACK_MESSAGE,
        "retrieved_docs": state.get("retrieved_docs", []),
    }


# create the graph
graph_builder = StateGraph(GuardrailedRAGState)

graph_builder.add_node("input_guardrail", input_guardrail_node)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("retrieval_guardrail", retrieval_guardrail_node)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)
graph_builder.add_node("output_guardrail", output_guardrail_node)
graph_builder.add_node("error_fallback", error_fallback_node)
graph_builder.add_node("soft_fallback", soft_fallback_node)

graph_builder.add_edge(START, "input_guardrail")

graph_builder.add_conditional_edges(
    "input_guardrail",
    guardrail_router,
    {"ok": "retrieve", "exception": "error_fallback", "refrain": "soft_fallback"},
)

graph_builder.add_edge("retrieve", "retrieval_guardrail")

graph_builder.add_conditional_edges(
    "retrieval_guardrail",
    guardrail_router,
    {"ok": "augmentation", "exception": "error_fallback", "refrain": "soft_fallback"},
)

graph_builder.add_edge("augmentation", "generation")
graph_builder.add_edge("generation", "output_guardrail")

graph_builder.add_conditional_edges(
    "output_guardrail",
    guardrail_router,
    {"ok": END, "exception": "error_fallback", "refrain": "soft_fallback"},
)

graph_builder.add_edge("error_fallback", END)
graph_builder.add_edge("soft_fallback", END)

graph = graph_builder.compile()
