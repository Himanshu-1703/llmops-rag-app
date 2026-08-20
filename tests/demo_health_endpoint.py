"""Manual demo: calls GET /health/dependencies and prints retriever + generator status.

Not a pytest test (no assertions, needs a live server) -- named `demo_*`
instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_health_endpoint.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"


def demo_dependency_health() -> None:
    response = requests.get(f"{BASE_URL}/health/dependencies")
    response.raise_for_status()
    status = response.json()

    print(f"Retriever : {status['retriever_health'].upper()}")
    if status["retriever_error"]:
        print(f"            error: {status['retriever_error']}")

    print(f"Generator : {status['llm_health'].upper()}")
    if status["llm_error"]:
        print(f"            error: {status['llm_error']}")

    print(f"\nOverall   : {status['overall_health'].upper()}")


if __name__ == "__main__":
    demo_dependency_health()
