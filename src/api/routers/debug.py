from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.documents import Document

from app.clients import logger
from app.rag_workflow import graph as plain_graph
from app.rag_workflow_with_guardrails import graph as guardrailed_graph

from api.schemas import (
    ChatRequest,
    DebugGuardrailedRAGStateResponse,
    DebugRAGStateResponse,
    SerializedDocument,
)
from api.security import require_admin_key

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
    # Runs the guardrailed graph (the one the /chat endpoint serves) and returns
    # its complete final state, including the guardrail status/stage/message.
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
