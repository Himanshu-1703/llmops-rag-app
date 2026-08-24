import os
from collections.abc import Iterator

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def stream_chat(query: str) -> Iterator[str]:
    with requests.post(
        f"{API_BASE_URL}/chat", json={"query": query}, stream=True
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def check_dependency_health() -> dict:
    # The retriever check runs a contextual-compression LLM extraction pass,
    # which can take tens of seconds -- give it real headroom.
    response = requests.get(f"{API_BASE_URL}/health/dependencies", timeout=90)
    response.raise_for_status()
    return response.json()


def list_uploaded_files() -> dict:
    response = requests.get(f"{API_BASE_URL}/chat/files", timeout=15)
    response.raise_for_status()
    return response.json()
