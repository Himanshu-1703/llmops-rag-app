import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    # x_admin_key defaults to None (rather than being a required Header)
    # so a missing header falls through to this check and returns a clean
    # 401, instead of FastAPI's automatic 422 for a missing required field.
    expected_key = os.environ.get("ADMIN_API_KEY")
    if not expected_key or not x_admin_key or x_admin_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
        )
