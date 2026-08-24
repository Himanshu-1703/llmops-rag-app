import re
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.rag_workflow_with_guardrails import graph
from app.vector_store import app_params, vs

from api.schemas import ChatRequest, UploadedFilesResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def stream_response(text: str) -> Iterator[str]:
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


@router.post("")
def chat(request: ChatRequest) -> StreamingResponse:
    # Input/retrieval/output guardrails run as nodes inside the graph
    # itself (see app/rag_workflow_with_guardrails.py). On a guardrail
    # failure, result["response"] is a fallback message set by
    # error_fallback_node/soft_fallback_node -- this route does not need
    # to branch on guardrail status.
    result = graph.invoke({"query": request.query})
    response_text = result["response"]
    return StreamingResponse(stream_response(response_text), media_type="text/plain")


def natural_sort_key(filename: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


@router.get("/files", response_model=UploadedFilesResponse)
def list_uploaded_files() -> UploadedFilesResponse:
    records = vs.get(include=["metadatas"])
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
