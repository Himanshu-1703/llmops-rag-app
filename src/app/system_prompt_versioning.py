from langfuse import get_client
from dotenv import load_dotenv

# load the api keys
load_dotenv()

chunk_size = 300
chunk_overlap = 30
output_dimensions = 1024
k = 3

system_prompt = """You are a helpful assistant. Answer the user query
based on the given context only. If you do not know the answer
say I don't know. Do not add any preamble to the response"""

# add system prompt to langfuse

langfuse = get_client()

created_prompt = langfuse.create_prompt(
    name="rag_app_system_prompt",
    type="text",
    prompt=system_prompt,
    labels=["baseline"],
    config={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_dims": output_dimensions,
        "k": k
    }
)

print(created_prompt.prompt)
print(created_prompt.version)
print(created_prompt.labels)