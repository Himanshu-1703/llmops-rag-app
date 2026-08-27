"""Manual demo: calls POST /internal/debug/rag and prints the whole final RAG state.

Hits the *plain* RAG graph (retrieve -> augmentation -> generation, no guardrails)
via the internal debug router and pretty-prints its complete final state --
query, retrieved docs, built context, prompt, and response. No streaming.

Not a pytest test (no assertions, needs a live server) -- named `demo_*`
instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_rag_app.py
       (reads ADMIN_API_KEY from .env, then prompts for a query)
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-Admin-Key": os.environ.get("ADMIN_API_KEY")}
# Contextual-compression retrieval can take up to ~90s (see CLAUDE.md).
REQUEST_TIMEOUT = 120


def print_rag_state(state: dict) -> None:
    print("\n===== FINAL RAG STATE (raw JSON) =====")
    print(json.dumps(state, indent=2, ensure_ascii=False))

    print("\n===== FINAL RAG STATE (summary) =====")
    print(f"query    : {state.get('query')}")

    docs = state.get("retrieved_docs") or []
    print(f"\nretrieved_docs ({len(docs)}):")
    for i, doc in enumerate(docs):
        preview = doc.get("page_content", "").replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"  [{i}] metadata: {doc.get('metadata')}")
        print(f"      content : {preview}")

    print(f"\ncontext  :\n{state.get('context')}")
    print(f"\nprompt   :\n{state.get('prompt')}")
    print(f"\nresponse :\n{state.get('response')}")


def demo_rag_app(query: str) -> None:
    print(f"Query: {query}")

    response = requests.post(
        f"{BASE_URL}/internal/debug/rag",
        json={"query": query},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    print_rag_state(response.json())


if __name__ == "__main__":
    user_query = input("Enter your query: ").strip()
    demo_rag_app(user_query)
