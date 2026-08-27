from typing import Literal

from pydantic import BaseModel, computed_field


class ChunkCountResponse(BaseModel):
    collection: str
    count: int


class TranscriptSyncResponse(BaseModel):
    files_scanned: int
    files_ingested: int
    chunks_added: int
    skipped_files: list[str]


class AppHealthResponse(BaseModel):
    message: str


class DependencyHealthResponse(BaseModel):
    llm_health: Literal["healthy", "unhealthy"]
    llm_error: str | None = None
    retriever_health: Literal["healthy", "unhealthy"]
    retriever_error: str | None = None

    @computed_field
    @property
    def overall_health(self) -> Literal["healthy", "unhealthy"]:
        if self.llm_health == "healthy" and self.retriever_health == "healthy":
            return "healthy"
        else:
            return "unhealthy"


class ChatRequest(BaseModel):
    query: str


class SerializedDocument(BaseModel):
    page_content: str
    metadata: dict


class DebugRAGStateResponse(BaseModel):
    # Every field beyond `query` is optional: on guardrail fallback paths the
    # graph short-circuits and never populates some keys (e.g. an input-refrain
    # never runs retrieve/augmentation, so there are no docs/context/prompt).
    query: str
    retrieved_docs: list[SerializedDocument] = []
    context: str | None = None
    prompt: str | None = None  # str(ChatPromptTemplate) -- not JSON-native otherwise
    response: str | None = None


class DebugGuardrailedRAGStateResponse(DebugRAGStateResponse):
    guardrail_status: str | None = None
    guardrail_stage: str | None = None
    guardrail_message: str | None = None


class UploadedFilesResponse(BaseModel):
    collection: str
    unique_files: int
    filenames: list[str]
