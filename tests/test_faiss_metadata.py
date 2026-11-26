import json
import pytest
from src.retrieval.faiss_retriever import load_records

def test_metadata_mapping(tmp_path):
    a={"chunk_id":"a","text":"Example evidence A"}
    b={"chunk_id":"b","text":"Example evidence B"}
    m=tmp_path/"metadata.json"; c=tmp_path/"chunks.jsonl"
    m.write_text(json.dumps([b,a]),encoding="utf-8")
    c.write_text("\n".join(map(json.dumps,[a,b])),encoding="utf-8")
    assert [r["chunk_id"] for r in load_records(m,c)]==["b","a"]
    m.write_text(json.dumps([{**b,"text":"Changed"},a]),encoding="utf-8")
    with pytest.raises(ValueError,match="text differs"):
        load_records(m,c)

def test_duplicate_mapping(tmp_path):
    a={"chunk_id":"a","text":"Example evidence A"}
    m=tmp_path/"metadata.json"; c=tmp_path/"chunks.jsonl"
    m.write_text(json.dumps([a,a]),encoding="utf-8")
    c.write_text(json.dumps(a),encoding="utf-8")
    with pytest.raises(ValueError,match="Duplicate"):
        load_records(m,c)
