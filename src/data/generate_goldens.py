from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, ContextConstructionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path
from config.parameter_config import params_config

# load golden dataset params
golden_params = params_config.golden_dataset

# load the api keys
load_dotenv()

def generate_golden_dataset():

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


    # load the config parameters
    filtration_params = golden_params.filtration_config
    context_construction_params = golden_params.context_construction_config
    evolution_params = golden_params.evolution_config

    # create the configs
    filtration_config = FiltrationConfig(
        synthetic_input_quality_threshold=filtration_params.filtration_threshold,
        max_quality_retries=filtration_params.max_quality_retries,
        critic_model=filtration_params.filtration_critic_model,
    )

    evolution_config = EvolutionConfig(
        num_evolutions=evolution_params.num_evolutions,
        evolutions={
            Evolution.MULTICONTEXT: evolution_params.evolutions["multicontext"],
            Evolution.CONCRETIZING: evolution_params.evolutions["concretizing"],
            Evolution.CONSTRAINED: evolution_params.evolutions["constrained"],
            Evolution.COMPARATIVE: evolution_params.evolutions["comparative"]
        }
    )

    context_config = ContextConstructionConfig(
        critic_model=context_construction_params.context_critic_model,
        max_contexts_per_document=context_construction_params.max_contexts_per_document,
        context_quality_threshold=context_construction_params.context_threshold,
        max_retries=context_construction_params.max_context_retries,
        chunk_size=context_construction_params.context_chunk_size,
        chunk_overlap=context_construction_params.context_chunk_overlap
    )

    # create the synthesizer
    synthesizer = Synthesizer(
        model=golden_params.dataset_model,
        filtration_config=filtration_config,
        evolution_config=evolution_config
    )

    # generate the goldens
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=get_docs_path(DOCS_PATH),
        max_goldens_per_context=golden_params.max_goldens_per_context,
        context_construction_config=context_config)

    # save the goldens
    synthesizer.save_as(
        file_type="json",
        directory=GOLDENS_DIR.as_posix(),
        file_name=golden_params.golden_dataset_filename
    )
    
if __name__ == "__main__":
    generate_golden_dataset()