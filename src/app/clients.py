from pathlib import Path
from logging import getLogger, StreamHandler, Formatter, INFO

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config.parameter_config import params_config

# load the api keys
load_dotenv()

# load the application parameters
app_params = params_config.rag_app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# create the logger
logger = getLogger(name="Rag App")
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
# don't hand records to the root logger too -- some deps (presidio) call
# logging.basicConfig(), which attaches a root handler, so without this every
# line is emitted twice: once by our handler, once by root's.
logger.propagate = False
formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)

# shared llm client -- used by the RAG graph's generation step, the vector
# store's contextual-compression retriever, and the /health dependency check
llm = ChatOpenAI(model=app_params.llm, temperature=0)
