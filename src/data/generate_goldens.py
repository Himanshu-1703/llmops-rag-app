from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, ContextConstructionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path

# load the api keys
load_dotenv()

# define paths
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DOCS_PATH = ROOT_PATH / "data" / "processed"
GOLDENS_DIR = ROOT_PATH / "data" / "evaluation" / "goldens"

# create the directory
GOLDENS_DIR.mkdir(exist_ok=True, parents=True)

def get_docs_path(directory_path: Path | str) -> list[str]:
    directory_path = Path(directory_path)
    if directory_path.exists() and directory_path.is_dir():
        paths = directory_path.glob("*.txt")
    return [path.as_posix() for path in paths]

# create the configs
filtration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.6,
    max_quality_retries=2,
    critic_model="gpt-5-mini",
)

evolution_config = EvolutionConfig(
    num_evolutions=2,
    evolutions={
        Evolution.MULTICONTEXT: 0.1,
        Evolution.CONCRETIZING: 0.3,
        Evolution.CONSTRAINED: 0.4,
        Evolution.COMPARATIVE: 0.2
    }
)

context_config = ContextConstructionConfig(
    critic_model="gpt-5.4-mini",
    max_contexts_per_document=1,
    context_quality_threshold=0.7,
    max_retries=2,
    chunk_size=300,
    chunk_overlap=30
)

# create the synthesizer
synthesizer = Synthesizer(
    model="gpt-5.4-mini",
    filtration_config=filtration_config,
    evolution_config=evolution_config
)

# generate the goldens
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=get_docs_path(DOCS_PATH),
    max_goldens_per_context=1,
    context_construction_config=context_config)

# save the goldens
synthesizer.save_as(
    file_type="json",
    directory=GOLDENS_DIR.as_posix(),
    file_name="golden_dataset_final"
)