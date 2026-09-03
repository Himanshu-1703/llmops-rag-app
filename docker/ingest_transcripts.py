"""Build-time vector-store construction (run inside docker/Dockerfile.api).

Imports only app.vector_store + utils.transcript_utils, so it needs OPENAI_API_KEY
but deliberately never imports the RAG graph / Langfuse. Cleans data/raw -> data/processed
and embeds every chunk into saved-embeddings/ using the models in params.yaml.
"""

from app.clients import REPO_ROOT
from app.vector_store import (
    PROCESSED_TRANSCRIPTS_DIR,
    chunk_overlap,
    chunk_size,
    upsert_documents,
)
from utils.transcript_utils import load_and_clean_transcripts

load_and_clean_transcripts(REPO_ROOT / "data" / "raw", PROCESSED_TRANSCRIPTS_DIR)
print("ingest:", upsert_documents(chunk_size, chunk_overlap))
