from pydantic import BaseModel


class RagAppConfig(BaseModel):
    
    prompt_label: str
    llm: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    tokenizer_encoding: str
    collection_name: str
    search_type: str
    k: int
    
    model_config = {"extra": "forbid"}
    

class FiltrationConfig(BaseModel):
    
    filtration_threshold: float
    max_quality_retries: int
    filtration_critic_model: str
    
    model_config = {"extra": "forbid"}


class EvolutionConfig(BaseModel):
    
    num_evolutions: int
    evolutions: dict[str, float]
    
    model_config = {"extra": "forbid"}


class ContextConstructionConfig(BaseModel):
    
    context_critic_model: str
    max_contexts_per_document: int
    context_threshold: float
    max_context_retries: int
    context_chunk_size: int
    context_chunk_overlap: int
    
    model_config = {"extra": "forbid"}


class GoldenDatasetConfig(BaseModel):
    
    dataset_model: str
    max_goldens_per_context: int
    golden_dataset_filename: str
    filtration_config: FiltrationConfig
    evolution_config: EvolutionConfig
    context_construction_config: ContextConstructionConfig
    
    model_config = {"extra": "forbid"}


class EvaluationDatasetConfig(BaseModel):
    
    evaluation_dataset_filename: str
    
    model_config = {"extra": "forbid"}
   
    
class AsyncConfig(BaseModel):
    
    throttle_value: int
    max_concurrent: int
    
    model_config = {"extra": "forbid"}


class DisplayConfig(BaseModel):
    
    results_dir: str
    report_dir: str
    
    model_config = {"extra": "forbid"}
    
    
class EvaluationConfig(BaseModel):
    
    judge_llm: str
    async_config: AsyncConfig
    display_config: DisplayConfig
    
    
    model_config = {"extra": "forbid"}


class Params(BaseModel):
    
    rag_app: RagAppConfig
    golden_dataset: GoldenDatasetConfig
    evaluation_dataset: EvaluationDatasetConfig
    evaluation: EvaluationConfig
    
    model_config = {"extra": "forbid"}