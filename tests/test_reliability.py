from src.retrieval.lexical import BM25, is_reference_chunk, reciprocal_rank_fusion
from src.generation.generator import generation_hit_limit

def test_bm25_and_fusion():
    ranker=BM25(["clinical symptoms of diabetes","annual report and bibliography","diabetes glucose"])
    assert ranker.scores("clinical symptoms")[0]>ranker.scores("clinical symptoms")[2]
    assert reciprocal_rank_fusion([[1,2,3],[2,4,1]])[0] in {1,2}

def test_reference_filter():
    assert is_reference_chunk({"clinical_section":"References","text":"example"})
    assert is_reference_chunk({"text":"Bibliography\nA source"})
    assert not is_reference_chunk({"clinical_section":"Diagnosis","text":"A glucose test measures blood glucose."})

def test_generation_budget():
    assert generation_hit_limit([1,2,3],3,[9])
    assert not generation_hit_limit([1,2,9],3,[9])
    assert not generation_hit_limit([1,2],3,[9])
