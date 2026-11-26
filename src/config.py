from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "dev"
    db_path: Path = Path("outputs/platform.sqlite")
    kb_dir: Path = Path("data/knowledge_base")
    index_dir: Path = Path("artifacts/index")
    model_registry_path: Path = Path("artifacts/model_registry.json")
    generation_backend: str = "extractive"
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_adapter_path: str = ""
    faiss_index_path: Path = Path("artifacts/index/guidelines.faiss")
    faiss_metadata_path: Path = Path("artifacts/index/metadata.json")
    faiss_build_info_path: Path = Path("artifacts/index/build_info.json")
    guideline_chunks_path: Path = Path("data/knowledge_base/guideline_chunks.jsonl")
    embedding_device: str = "cpu"
    embedding_backend: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_backend: str = "heuristic"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_candidates: int = 20
    top_k_final: int = 5
    retrieval_gate: float = 0.60
    tfidf_retrieval_gate: float = 0.15
    base_confidence_threshold: float = 0.68

SETTINGS = Settings()
