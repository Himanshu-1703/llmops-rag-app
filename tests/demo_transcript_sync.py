"""Manual demo: calls POST /internal/vector-store/transcripts/sync and prints ingestion stats.

Two-step manual demo (done by hand, this script just runs the sync each time):
    1. Delete `saved-embeddings/`, run this file -> syncs all transcripts from scratch.
    2. Drop 1-2 new transcripts into `data/raw/`, run this file again -> only the new
       files' chunks get embedded and added (existing ones are skipped via content hash).

Not a pytest test (no assertions, needs a live server, mutates the vector store) --
named `demo_*` instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_transcript_sync.py
       (reads ADMIN_API_KEY from .env)
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-Admin-Key": os.environ.get("ADMIN_API_KEY")}


def get_chunk_count() -> int:
    response = requests.get(f"{BASE_URL}/internal/vector-store/chunks/count", headers=HEADERS)
    response.raise_for_status()
    return response.json()["count"]


def demo_transcript_sync() -> None:
    before_count = get_chunk_count()
    print(f"Chunks in vector store before sync: {before_count}")

    response = requests.post(
        f"{BASE_URL}/internal/vector-store/transcripts/sync", headers=HEADERS
    )
    response.raise_for_status()
    stats = response.json()

    print("\nSync results:")
    print(f"  files_scanned  : {stats['files_scanned']}")
    print(f"  files_ingested : {stats['files_ingested']}")
    print(f"  chunks_added   : {stats['chunks_added']}")
    print(f"  skipped_files  : {stats['skipped_files']}")

    after_count = get_chunk_count()
    print(f"\nChunks in vector store after sync: {after_count}")
    print(f"Net new chunks added: {after_count - before_count}")


if __name__ == "__main__":
    demo_transcript_sync()
