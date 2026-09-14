import re
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.clients import logger
from app.rag_workflow_with_guardrails import graph
from app.vector_store import app_params, vs

from api.schemas import ChatRequest, UploadedFilesResponse

router = APIRouter(prefix="/chat", tags=["chat"])

langfuse = get_client()
langfuse_handler = CallbackHandler()


def stream_response(text: str) -> Iterator[str]:
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


@router.post("")
def chat(request: ChatRequest) -> StreamingResponse:
    # Serves the guardrailed graph (app/rag_workflow_with_guardrails.py):
    # input/retrieval/output guardrail nodes wrapped around retrieve ->
    # augmentation -> generation, with canned fallback messages on a trip.
    # The plain graph (app/rag_workflow.py) is kept for evaluation only.
    #
    # Root span sets explicit input/output for Langfuse instead of letting
    # the raw graph state become the trace I/O -- that state carries
    # non-serializable objects (e.g. ChatPromptTemplate) and internal fields
    # no one needs to see at the trace level.
    with langfuse.start_as_current_observation(
        as_type="span",
        name="chat-response",
        input=request.query,
    ) as root_span:
        with propagate_attributes(
            session_id=request.session_id,
            trace_name="chat-response",
        ):
            try:
                result = graph.invoke(
                    {"query": request.query},
                    config={"callbacks": [langfuse_handler], "run_name": "rag-graph"},
                )
            except Exception as e:
                logger.error(f"[chat] graph invocation failed: {e}")
                root_span.update(level="ERROR", status_message=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to process chat request.",
                ) from e
        response_text = result["response"]
        root_span.update(output=response_text)
    return StreamingResponse(stream_response(response_text), media_type="text/plain")


def natural_sort_key(filename: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


@router.get("/files", response_model=UploadedFilesResponse)
def list_uploaded_files() -> UploadedFilesResponse:
    try:
        records = vs.get(include=["metadatas"])
    except Exception as e:
        logger.error(f"[chat] failed to read vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve uploaded files.",
        ) from e
    filenames = sorted(
        {
            Path(meta["source"]).name
            for meta in records["metadatas"]
            if meta and meta.get("source")
        },
        key=natural_sort_key,
    )
    return UploadedFilesResponse(
        collection=app_params.collection_name,
        unique_files=len(filenames),
        filenames=filenames,
    )
