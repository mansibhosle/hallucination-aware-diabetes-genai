from pathlib import Path
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import GroundedGenerator
from src.pipeline import Platform

class S:
    top_k_candidates=20; top_k_final=5; retrieval_gate=.10; base_confidence_threshold=.40
    db_path=Path("/tmp/test_diabetes_platform.sqlite")

def test_pipeline_answer_and_abstain():
    chunks=[{"chunk_id":"1","source_id":"demo","title":"Prediabetes","text":"Prediabetes means blood glucose levels are higher than normal but not yet in the diabetes range.","sha256":"1"}]
    p=Platform(HybridRetriever().fit(chunks),Reranker(),GroundedGenerator(),S())
    assert p.answer("What is prediabetes?")["decision"] in {"ANSWER","ABSTAIN_CONFIDENCE"}
    assert p.answer("How many units of insulin should I take?")["decision"].startswith("ABSTAIN")
