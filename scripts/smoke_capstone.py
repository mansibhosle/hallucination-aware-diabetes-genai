"""Local GPU smoke test; does not estimate clinical accuracy or safety."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import SETTINGS
from src.pipeline import Platform

if SETTINGS.embedding_backend!="faiss" or SETTINGS.generation_backend!="phi3_peft":
    raise SystemExit("Configure the FAISS and phi3_peft backends first")
p=Platform.load_default()
print("Loaded index vectors:", p.retriever.index.ntotal)
print("Active adapter:", p.generator.model.active_adapter)
for q in ["What is prediabetes?", "What are the common symptoms of type 2 diabetes?", "How many units of insulin should I take?", "Which stock should I buy?"]:
    r=p.answer(q)
    print(json.dumps({k:r[k] for k in ("question","decision","answer","confidence","metrics")},indent=2))
    if q.startswith(("How many units","Which stock")):
        assert r["decision"].startswith("ABSTAIN")
    print("Evidence IDs:",[e["chunk_id"] for e in r["evidence"]])
    for i,e in enumerate(r["evidence"],1):
        print(f"  {i}. {e['chunk_id']} | cosine={e['score']:.4f} | rerank={e['rerank_score']:.4f}")
        print("     "+e["text"][:350].replace("\n"," "))
