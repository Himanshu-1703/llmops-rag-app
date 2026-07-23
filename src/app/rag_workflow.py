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


# load the api keys
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TRANSCRIPTS_DIR = REPO_ROOT / "data" / "processed"
VECTOR_STORE_DIR = REPO_ROOT / "saved-embeddings"

# create the llm and embedding model
llm = ChatOpenAI(model="gpt-5-mini")

embedder = OpenAIEmbeddings(model="text-embedding-3-small",
                            dimensions=1024) # 1536


chunk_size = 300
chunk_overlap = 30

# load and update knowledge base
def upsert_documents(chunk_size: int, chunk_overlap: int):
    # load the docs from source
    loader = DirectoryLoader(path=PROCESSED_TRANSCRIPTS_DIR.as_posix(),
                         loader_cls=TextLoader,
                         show_progress=True)
    
    # load the docs
    docs = loader.load()

    # chunk the text, sized in tokens (o200k_base matches the gpt-5 model family)
    # to mirror DeepEval's ContextConstructionConfig chunk sizing
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    chunks = chunker.split_documents(docs)


    # vector store
    vs = Chroma(collection_name="rag_demo",
            embedding_function=embedder,
            persist_directory=VECTOR_STORE_DIR.as_posix())

    # add docs
    vs.add_documents(chunks)

    return vs


def load_knowledge_base():
    # vector store
    vs = Chroma(collection_name="rag_demo",
            embedding_function=embedder,
            persist_directory=VECTOR_STORE_DIR.as_posix())


    return vs


if VECTOR_STORE_DIR.exists():
    vs = load_knowledge_base()
else:
    vs = upsert_documents(chunk_size, chunk_overlap)

# create the retriever
retriever = vs.as_retriever(search_type="similarity",
                            search_kwargs={"k":3})


class RAGState(TypedDict):
    
    query: str 
    retrieved_docs: list[Document]
    context: str 
    prompt: ChatPromptTemplate
    response: str


def retrieve(state: RAGState) -> dict:
    query = state["query"]
    retrieved_docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs":retrieved_docs,
            "context":context}



def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user query
                      based on the given context only. If you do not know the answer
                      say I don't know. Do not add any preamble to the response"""),
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