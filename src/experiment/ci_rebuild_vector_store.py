"""CI-only: rebuild the Chroma vector store from scratch against the current params.yaml.

Not wired into the local dev workflow -- the CI pipeline is the only caller:

    uv run python src/experiment/ci_rebuild_vector_store.py

The `saved-embeddings/` checked into git was embedded under a *previous*
configuration (embedding model / dimensions / chunk sizing). A CI run triggered
by a `params.yaml` change must re-embed the corpus before the experiment
pipeline runs, otherwise retrieval is scored against a stale collection.

This wipes `saved-embeddings/` and re-ingests `data/raw/` -> clean -> chunk ->
embed -> persist, reusing the same code path as the admin `/vector-store/sync`
endpoint (`load_and_clean_transcripts` + `upsert_documents`). The directory is
deleted *before* `app.vector_store` is imported, because that module opens the
Chroma connection at import time.
"""
import logging
import shutil
from logging import INFO
from pathlib import Path

from dotenv import load_dotenv

# load the api keys (OPENAI_API_KEY -- needed to embed)
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = ROOT_DIR / "saved-embeddings"

logger = logging.getLogger(name="CI vector store builder")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)


def main() -> None:
    # wipe the stale collection BEFORE importing app.vector_store (that module
    # opens a persistent Chroma client against this path on import).
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
        logger.info(f"Removed stale vector store at {VECTOR_STORE_DIR}")
    else:
        logger.info("No existing vector store to remove")

    from app.vector_store import (
        PROCESSED_TRANSCRIPTS_DIR,
        chunk_overlap,
        chunk_size,
        upsert_documents,
    )
    from utils.transcript_utils import load_and_clean_transcripts

    RAW_TRANSCRIPTS_DIR = ROOT_DIR / "data" / "raw"

    load_and_clean_transcripts(RAW_TRANSCRIPTS_DIR, PROCESSED_TRANSCRIPTS_DIR)
    logger.info("Transcripts cleaned")

    stats = upsert_documents(chunk_size, chunk_overlap)
    logger.info(f"Vector store rebuilt: {stats}")


if __name__ == "__main__":
    main()
