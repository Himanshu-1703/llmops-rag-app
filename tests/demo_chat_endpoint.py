"""Manual demo: calls POST /chat and prints the response as it streams in.

Not a pytest test (no assertions, needs a live server) -- named `demo_*`
instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_chat_endpoint.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
QUERY = "What is the difference between online and offline evals"


def demo_chat_streaming(query: str) -> None:
    print(f"Query: {query}\n")
    print("Response: ", end="", flush=True)

    with requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        stream=True,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    demo_chat_streaming(QUERY)
