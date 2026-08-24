"""Manual demo: calls GET /chat/files and prints the indexed source filenames.

Not a pytest test (no assertions, needs a live server) -- named `demo_*`
instead of `test_*` so pytest doesn't try to collect and run it.

Usage:
    1. Start the API:  uvicorn api.main:app --reload --app-dir src
    2. Run this file:  python tests/demo_uploaded_files.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"


def demo_uploaded_files() -> None:
    response = requests.get(f"{BASE_URL}/chat/files")
    response.raise_for_status()
    data = response.json()

    print(f"Collection   : {data['collection']}")
    print(f"Unique files : {data['unique_files']}\n")
    for filename in data["filenames"]:
        print(f"  - {filename}")


if __name__ == "__main__":
    demo_uploaded_files()
