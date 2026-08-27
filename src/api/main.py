from fastapi import FastAPI

from api.routers import chat, debug, health, vector_store

app = FastAPI(title="RAG App - Internal API")

app.include_router(chat.router)
app.include_router(health.router)
app.include_router(vector_store.router)
app.include_router(debug.router)
