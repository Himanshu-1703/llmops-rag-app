from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client
from langgraph.graph import END, START, StateGraph

from app.clients import app_params, llm
from app.vector_store import get_retriever

# langfuse client
langfuse = get_client()

# load system prompt
system_prompt = langfuse.get_prompt(
    name="rag_app_system_prompt",
    type="text",
    label=app_params.prompt_label
)


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
