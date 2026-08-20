from fastapi import APIRouter

from app.clients import llm
from app.vector_store import get_retriever

from api.schemas import AppHealthResponse, DependencyHealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=AppHealthResponse)
def application_health() -> AppHealthResponse:
    return AppHealthResponse(message="welcome to campusx chatbot")


@router.get("/dependencies", response_model=DependencyHealthResponse)
def dependency_health() -> DependencyHealthResponse:
    llm_health: str = "healthy"
    llm_error: str | None = None
    try:
        llm.invoke("Hi")
    except Exception as e:
        llm_health = "unhealthy"
        llm_error = str(e)

    retriever_health: str = "healthy"
    retriever_error: str | None = None
    try:
        docs = get_retriever().invoke("What is contextual recall")
        if not docs:
            retriever_health = "unhealthy"
            retriever_error = "No documents returned"
    except Exception as e:
        retriever_health = "unhealthy"
        retriever_error = str(e)

    return DependencyHealthResponse(
        llm_health=llm_health,
        llm_error=llm_error,
        retriever_health=retriever_health,
        retriever_error=retriever_error,
    )
