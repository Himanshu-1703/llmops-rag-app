from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.documents import Document

from app.clients import logger
from app.rag_workflow import graph as plain_graph

from api.schemas import (
    ChatRequest,
    DebugGuardrailedRAGStateResponse,
    DebugRAGStateResponse,
    SerializedDocument,
)
from api.security import require_admin_key

try:
    # The `guardrails` dependency group (guardrails-ai, torch, transformers, ...)
    # is optional -- docker/Dockerfile.api builds the API image with
    # --no-default-groups and doesn't install it, so this module is absent in
    # that deployment. Degrade the /rag_with_guardrails endpoint to a 503
    # instead of crashing the whole app at import time.
    from app.rag_workflow_with_guardrails import graph as guardrailed_graph
except ModuleNotFoundError:
    guardrailed_graph = None

router = APIRouter(
    prefix="/internal/debug",
    tags=["debug"],
    dependencies=[Depends(require_admin_key)],
)


def _serialize_docs(docs: list[Document] | None) -> list[SerializedDocument]:
    return [
        SerializedDocument(page_content=d.page_content, metadata=d.metadata)
        for d in (docs or [])
    ]


def _common_state(state: dict) -> dict:
    # Pull the shared RAG-state keys out of whatever the graph produced.
    # Nodes that didn't run simply leave their keys absent -> None / [].
    prompt = state.get("prompt")
    return {
        "query": state.get("query", ""),
        "retrieved_docs": _serialize_docs(state.get("retrieved_docs")),
        "context": state.get("context"),
        "prompt": str(prompt) if prompt is not None else None,
        "response": state.get("response"),
    }


@router.post("/rag", response_model=DebugRAGStateResponse)
def debug_rag(request: ChatRequest) -> DebugRAGStateResponse:
    # Runs the plain retrieve -> augmentation -> generation graph (no guardrails)
    # and returns its complete final state as a validated, non-streamed payload.
    try:
        state = plain_graph.invoke({"query": request.query})
    except Exception as e:
        logger.error(f"[debug] plain graph invocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run the RAG graph.",
        ) from e
    return DebugRAGStateResponse(**_common_state(state))


@router.post("/rag_with_guardrails", response_model=DebugGuardrailedRAGStateResponse)
def debug_rag_with_guardrails(
    request: ChatRequest,
) -> DebugGuardrailedRAGStateResponse:
    # Runs the guardrailed graph and returns its complete final state,
    # including the guardrail status/stage/message. Not wired into /chat --
    # see docker/Dockerfile.api and app/rag_workflow_with_guardrails.py.
    if guardrailed_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guardrails dependencies are not installed in this deployment.",
        )
    try:
        state = guardrailed_graph.invoke({"query": request.query})
    except Exception as e:
        logger.error(f"[debug] guardrailed graph invocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run the guardrailed RAG graph.",
        ) from e
    return DebugGuardrailedRAGStateResponse(
        **_common_state(state),
        guardrail_status=state.get("guardrail_status"),
        guardrail_stage=state.get("guardrail_stage"),
        guardrail_message=state.get("guardrail_message"),
    )
