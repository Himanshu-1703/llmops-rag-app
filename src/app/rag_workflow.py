from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from config.parameter_config import params_config
from logging import getLogger, StreamHandler, Formatter, INFO
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# load the application parameters
app_params = params_config.rag_app

# load the api keys
load_dotenv()

# langfuse client
langfuse = get_client()

# load system prompt
system_prompt = langfuse.get_prompt(
    name="rag_app_system_prompt",
    type="text",
    label=app_params.prompt_label
)

# create the logger
logger = getLogger(name="Rag App")
# add stream handler
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
# add formatter
formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TRANSCRIPTS_DIR = REPO_ROOT / "data" / "processed"
VECTOR_STORE_DIR = REPO_ROOT / "saved-embeddings"

# create the llm and embedding model
llm = ChatOpenAI(model=app_params.llm)

embedder = OpenAIEmbeddings(model=app_params.embedding_model,
                            dimensions=app_params.embedding_dimensions) # 1536


chunk_size = app_params.chunk_size
chunk_overlap = app_params.chunk_overlap

logger.info(f"Chunk Size: {chunk_size}")
logger.info(f"Chunk Overlap: {chunk_overlap}")
logger.info(f"Embedding Dimensions: {app_params.embedding_dimensions}")

# load and update knowledge base
def upsert_documents(chunk_size: int, chunk_overlap: int):
    # load the docs from source
    loader = DirectoryLoader(path=PROCESSED_TRANSCRIPTS_DIR.as_posix(),
                         loader_cls=TextLoader,
                         show_progress=True)
    
    # load the docs
    docs = loader.load()
    logger.info(f"Number of Transcripts Loaded: {len(docs)}")

    # chunk the text, sized in tokens (o200k_base matches the gpt-5 model family)
    # to mirror DeepEval's ContextConstructionConfig chunk sizing
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=app_params.tokenizer_encoding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    chunks = chunker.split_documents(docs)
    logger.info(f"Number of Chunks Created: {len(chunks)}")


    # vector store
    vs = Chroma(collection_name=app_params.collection_name,
            embedding_function=embedder,
            persist_directory=VECTOR_STORE_DIR.as_posix())

    # add docs
    doc_ids = vs.add_documents(chunks)
    logger.info(f"Number of Chunks added to Vector Database: {len(doc_ids)}")
    
    return vs


def load_knowledge_base():
    # vector store
    vs = Chroma(collection_name=app_params.collection_name,
            embedding_function=embedder,
            persist_directory=VECTOR_STORE_DIR.as_posix())


    return vs


if VECTOR_STORE_DIR.exists():
    vs = load_knowledge_base()
else:
    vs = upsert_documents(chunk_size, chunk_overlap)


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


class RAGState(TypedDict):
    
    query: str 
    retrieved_docs: list[Document]
    context: str 
    prompt: ChatPromptTemplate
    response: str


def retrieve(state: RAGState) -> dict:
    query = state["query"]
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs":retrieved_docs,
            "context":context}



def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.prompt),
        ("human", "context: {context}\n\nquery: {query}")
    ])
    
    return {"prompt": prompt}



def generation(state: RAGState) -> dict:
    
    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]
    
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context":context, "query":query})
    
    return {"response":response}



# create the graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)

graph_builder.add_edge(START,"retrieve")
graph_builder.add_edge("retrieve","augmentation")
graph_builder.add_edge("augmentation","generation")
graph_builder.add_edge("generation",END)

graph = graph_builder.compile()