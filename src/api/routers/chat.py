import re
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.clients import logger
from app.rag_workflow import graph
from app.vector_store import app_params, vs

from api.schemas import ChatRequest, UploadedFilesResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def stream_response(text: str) -> Iterator[str]:
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


@router.post("")
def chat(request: ChatRequest) -> StreamingResponse:
    # Serves the plain retrieve -> augmentation -> generation graph
    # (app/rag_workflow.py). The guardrailed graph exists in the repo but is
    # not wired into this deployment; see docker/Dockerfile.api.
    try:
        result = graph.invoke({"query": request.query})
    except Exception as e:
        logger.error(f"[chat] graph invocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request.",
        ) from e
    response_text = result["response"]
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
