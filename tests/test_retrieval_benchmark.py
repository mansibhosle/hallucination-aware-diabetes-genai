import pandas as pd
import pytest
from src.evaluation.retrieval_benchmark import parse_source_ids, parse_bool, load_validation, source_metrics, aggregate

def test_labels():
    assert parse_source_ids('CDC_001; NIDDK_PDF_002|DIA_01')=={"CDC_001","NIDDK_PDF_002","DIA_01"}
    assert parse_source_ids('["CDC_001", "DIA_01"]')=={"CDC_001","DIA_01"}
    assert not parse_bool("False")
    assert parse_bool("1")
    with pytest.raises(ValueError):
        parse_bool("maybe")

def test_split_and_unmapped(tmp_path):
    rows=[
      ["Q1","a","general","False","A","validation"],
      ["Q2","b","general","True","","validation"],
      ["Q3","c","general","False","Z","validation"],
      ["Q4","d","general","False","A","test"]]
    path=tmp_path/"eval.csv"
    pd.DataFrame(rows,columns=["question_id","question","category","should_abstain","reference_source_ids","split"]).to_csv(path,index=False)
    cases,excluded,total=load_validation(path,{"A"})
    assert total==3 and [c["question_id"] for c in cases]==["Q1"]
    assert [r["reason"] for r in excluded]==["should_abstain","missing_or_unmapped_source_ids"]

def test_metrics():
    m=source_metrics({"A","B"},["X","A","A","X","X"])
    assert m=={"hit_at_1":0.0,"hit_at_5":1.0,"mrr_at_5":0.5,"source_recall_at_5":0.5}
    assert source_metrics({"A"},["X"]*5)["mrr_at_5"]==0
    result=aggregate([{**m,"candidate_source_hit":1.0,"latency_ms":100.0}])
    assert result["n"]==1 and result["mean_latency_ms"]==100.0
