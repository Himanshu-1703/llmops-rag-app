import hashlib

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from app.clients import app_params, logger, REPO_ROOT

PROCESSED_TRANSCRIPTS_DIR = REPO_ROOT / "data" / "processed"
VECTOR_STORE_DIR = REPO_ROOT / "saved-embeddings"

embedder = OpenAIEmbeddings(model=app_params.embedding_model,
                            dimensions=app_params.embedding_dimensions) # 1536

chunk_size = app_params.chunk_size
chunk_overlap = app_params.chunk_overlap

logger.info(f"Chunk Size: {chunk_size}")
logger.info(f"Chunk Overlap: {chunk_overlap}")
logger.info(f"Embedding Dimensions: {app_params.embedding_dimensions}")

# Opens the existing persisted collection at VECTOR_STORE_DIR if one
# exists, or creates an empty one otherwise. This only opens a connection
# -- no documents are read or embedded here. Ingestion only happens when
# upsert_documents() below is explicitly called, which is wired up to run
# only from the admin API (src/api/routers/vector_store.py), never as a
# side effect of importing this module or the RAG graph that depends on it.
vs = Chroma(collection_name=app_params.collection_name,
        embedding_function=embedder,
        persist_directory=VECTOR_STORE_DIR.as_posix())


def get_retriever():
    # create the retriever
    retriever = vs.as_retriever(search_type=app_params.search_type,
                                search_kwargs={"k":app_params.k})

    if app_params.contextual_compression:
        # compressor
        compression_llm = ChatOpenAI(model=app_params.compression_llm)
        compressor = LLMChainExtractor.from_llm(compression_llm)

        # compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        return compression_retriever

    return retriever


def chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# load and update knowledge base
def upsert_documents(chunk_size: int, chunk_overlap: int) -> dict:
    
    # chunk the text, sized in tokens (o200k_base matches the gpt-5 model family)
    # to mirror DeepEval's ContextConstructionConfig chunk sizing
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=app_params.tokenizer_encoding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    files = sorted(PROCESSED_TRANSCRIPTS_DIR.glob("*.txt"))
    skipped_files: list[str] = []
    to_embed_chunks: list[Document] = []
    to_embed_ids: list[str] = []

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not read {file_path.name}: {e}")
            skipped_files.append(file_path.name)
            continue

        doc = Document(page_content=text, metadata={"source": file_path.name})
        file_chunks = chunker.split_documents([doc])

        if not file_chunks:
            skipped_files.append(file_path.name)
            continue

        candidate_ids = [f"{file_path.name}::{i}" for i in range(len(file_chunks))]
        existing = vs.get(ids=candidate_ids, include=["metadatas"])
        existing_hash_by_id = {
            id_: (meta or {}).get("chunk_hash")
            for id_, meta in zip(existing["ids"], existing["metadatas"])
        }

        for chunk_id, chunk in zip(candidate_ids, file_chunks):
            new_hash = chunk_hash(chunk.page_content)
            if existing_hash_by_id.get(chunk_id) == new_hash:
                continue
            chunk.metadata["chunk_hash"] = new_hash
            to_embed_chunks.append(chunk)
            to_embed_ids.append(chunk_id)

    if to_embed_chunks:
        vs.add_documents(to_embed_chunks, ids=to_embed_ids)

    logger.info(
        f"Transcript upsert: scanned {len(files)} files, "
        f"embedded {len(to_embed_chunks)} new/changed chunks, "
        f"{len(skipped_files)} files unreadable/empty"
    )

    return {
        "files_scanned": len(files),
        "files_ingested": len(files) - len(skipped_files),
        "chunks_added": len(to_embed_chunks),
        "skipped_files": skipped_files,
    }
