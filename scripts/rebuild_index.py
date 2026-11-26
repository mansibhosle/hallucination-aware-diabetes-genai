from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.retrieval.hybrid import HybridRetriever
from src.data.versioning import build_manifest
from src.config import SETTINGS

files=list(Path("data/knowledge_base").glob("*.jsonl")); chunks=[]
for p in files:
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip(): chunks.append(json.loads(line))
if not chunks: raise SystemExit("No JSONL chunks found in data/knowledge_base")
HybridRetriever().fit(chunks).save(SETTINGS.index_dir/"hybrid.joblib")
print(build_manifest(chunks,SETTINGS.index_dir/"kb_manifest.json"))
