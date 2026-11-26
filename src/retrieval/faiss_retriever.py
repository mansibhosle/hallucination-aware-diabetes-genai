"""Load a locally owned, trusted capstone index; never download arbitrary indexes."""
import json
from pathlib import Path
from ..schemas import Evidence

def load_records(metadata_path, chunks_path):
    records = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty metadata list in FAISS row order")
    chunks = [json.loads(line) for line in Path(chunks_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(c["chunk_id"]): c for c in chunks}
    ids = [str(r["chunk_id"]) for r in records]
    if len(by_id) != len(chunks) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("Duplicate or mismatched chunk IDs")
    for record in records:
        original = by_id[str(record["chunk_id"])]
        if not record.get("text", "").strip() or record["text"] != original.get("text"):
            raise ValueError("Metadata text differs from original chunk: " + str(record["chunk_id"]))
        if record.get("text_sha256") and record["text_sha256"] != original.get("text_sha256"):
            raise ValueError("Metadata fingerprint mismatch")
    return records

class FaissRetriever:
    def __init__(self, index_path, metadata_path, build_info_path, chunks_path, embedding_model, device="cpu"):
        import faiss
        from sentence_transformers import SentenceTransformer

        info = json.loads(Path(build_info_path).read_text(encoding="utf-8"))
        if info.get("embedding_model") != embedding_model:
            raise ValueError("Embedding model differs from index build configuration")
        if "IndexFlatIP" not in info.get("index_type", "") or "normalized" not in info.get("index_type", "").lower():
            raise ValueError("Expected normalized IndexFlatIP embeddings")
        self.records = load_records(metadata_path, chunks_path)
        self.index = faiss.read_index(str(index_path))
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError("Expected FAISS IndexFlatIP")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected inner product")
        if self.index.ntotal != len(self.records) or info.get("chunk_count") != len(self.records):
            raise ValueError("Index and metadata counts differ")
        self.encoder = SentenceTransformer(embedding_model, device=device)
        if self.encoder.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Embedding dimensions do not match")
        self.verify_sample_mapping()

    def verify_sample_mapping(self, sample_size=5):
        import numpy as np
        rows = np.linspace(0, self.index.ntotal-1, min(sample_size,self.index.ntotal), dtype=int)
        vectors = self.encoder.encode([self.records[int(i)]["text"] for i in rows], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        for i, vec in zip(rows, vectors):
            stored = self.index.reconstruct(int(i))
            if not np.isclose(np.linalg.norm(stored),1.0,atol=0.01) or float(stored @ vec) < 0.98:
                raise ValueError(f"Sample embedding mismatch at row {i}; check index ordering and build configuration")

    def search(self, query, k=20):
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(q, min(k,self.index.ntotal))
        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0: continue
            record = self.records[int(idx)]
            results.append(Evidence(
                chunk_id=str(record["chunk_id"]), text=record["text"],
                source_id=str(record.get("source_id","")), title=record.get("document_title",""),
                url=record.get("source_url",""), score=float(score),
                metadata={k:v for k,v in record.items() if k!="text"},
            ))
        return results
