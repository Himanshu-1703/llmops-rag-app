from fastapi import APIRouter, Depends

from app.clients import REPO_ROOT, logger
from app.vector_store import (
    PROCESSED_TRANSCRIPTS_DIR,
    app_params,
    chunk_overlap,
    chunk_size,
    upsert_documents,
    vs,
)
from utils.transcript_utils import load_and_clean_transcripts

from api.schemas import ChunkCountResponse, TranscriptSyncResponse
from api.security import require_admin_key

router = APIRouter(
    prefix="/internal/vector-store",
    tags=["vector-store-admin"],
    dependencies=[Depends(require_admin_key)],
)

RAW_TRANSCRIPTS_DIR = REPO_ROOT / "data" / "raw"


@router.get("/chunks/count", response_model=ChunkCountResponse)
def count_chunks() -> ChunkCountResponse:
    # langchain_chroma.Chroma exposes no public count() wrapper; the
    # underlying chromadb Collection does, so we reach past the wrapper.
    count = vs._collection.count()
    return ChunkCountResponse(collection=app_params.collection_name, count=count)


@router.post("/transcripts/sync", response_model=TranscriptSyncResponse)
def sync_transcripts() -> TranscriptSyncResponse:
    load_and_clean_transcripts(RAW_TRANSCRIPTS_DIR, PROCESSED_TRANSCRIPTS_DIR)
    # The only place upsert_documents() gets called -- this endpoint is
    # the sole entry point for data-mutating ingestion. src/app/vector_store.py
    # never calls it on import, and neither does the RAG graph.
    stats = upsert_documents(chunk_size, chunk_overlap)
    return TranscriptSyncResponse(**stats)
