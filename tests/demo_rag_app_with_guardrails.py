"""Manual demo: calls POST /internal/debug/rag_with_guardrails and prints the whole state.

Hits the *guardrailed* RAG graph (the one /chat serves: input/retrieval/output
guardrail nodes + fallbacks) via the internal debug router and pretty-prints its
complete final state -- query, retrieved docs, context, prompt, response, plus the
guardrail status/stage/message. No streaming.

Not a pytest test (no assertions, needs a live server) -- named `demo_*`
instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_rag_app_with_guardrails.py
       (reads ADMIN_API_KEY from .env, then prompts for a query)
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-Admin-Key": os.environ.get("ADMIN_API_KEY")}
# Contextual-compression retrieval + 3 guardrail layers can take a while.
REQUEST_TIMEOUT = 120


def print_rag_state(state: dict) -> None:
    print("\n===== FINAL GUARDRAILED RAG STATE (raw JSON) =====")
    print(json.dumps(state, indent=2, ensure_ascii=False))

    print("\n===== FINAL GUARDRAILED RAG STATE (summary) =====")
    print(f"query            : {state.get('query')}")
    print(f"guardrail_status : {state.get('guardrail_status')}")
    print(f"guardrail_stage  : {state.get('guardrail_stage')}")
    print(f"guardrail_message: {state.get('guardrail_message')}")

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


def demo_rag_app_with_guardrails(query: str) -> None:
    print(f"Query: {query}")

    response = requests.post(
        f"{BASE_URL}/internal/debug/rag_with_guardrails",
        json={"query": query},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    print_rag_state(response.json())


if __name__ == "__main__":
    user_query = input("Enter your query: ").strip()
    demo_rag_app_with_guardrails(user_query)
