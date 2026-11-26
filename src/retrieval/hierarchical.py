"""Experimental child-window search with original parent evidence.

Builds a separate in-memory index. The original source index is never modified.
"""
import re
import numpy as np
from ..schemas import Evidence
from .lexical import BM25, is_reference_chunk, reciprocal_rank_fusion


def sentence_windows(text, max_words=80, overlap=20):
    """Return overlapping one/two-sentence windows, splitting long spans."""
    if not 0 <= overlap < max_words:
        raise ValueError("Invalid word window configuration")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    seen = set()
    for i, sentence in enumerate(sentences):
        spans = [sentence]
        if i + 1 < len(sentences):
            spans.append(sentence + " " + sentences[i + 1])
        for span in spans:
            words = span.split()
            for start in range(0, len(words), max_words - overlap):
                window = " ".join(words[start:start + max_words])
                if window and window not in seen:
                    seen.add(window)
                    yield window
                if start + max_words >= len(words):
                    break


def unique_parent_ranking(child_ids, parent_ids, limit):
    ranked = []
    seen = set()
    for child_id in child_ids:
        if child_id < 0:
            continue
        parent = int(parent_ids[int(child_id)])
        if parent not in seen:
            seen.add(parent)
            ranked.append(parent)
        if len(ranked) >= limit:
            break
    return ranked


class HierarchicalSearch:
    def __init__(self, records, encoder, parent_index):
        import faiss

        self.records = records
        self.encoder = encoder
        self.parent_index = parent_index
        self.windows = []
        parents = []
        for row, record in enumerate(records):
            if is_reference_chunk(record):
                continue
            for window in sentence_windows(record["text"]):
                self.windows.append(window)
                parents.append(row)
        if not self.windows:
            raise ValueError("No eligible child windows")
        self.parent_ids = np.asarray(parents, dtype=np.int64)
        self.bm25 = BM25(self.windows)
        self.index = faiss.IndexFlatIP(parent_index.d)
        for start in range(0, len(self.windows), 512):
            vectors = encoder.encode(
                self.windows[start:start + 512], batch_size=64,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            if vectors.ndim != 2 or vectors.shape[1] != parent_index.d:
                raise ValueError("Child embedding dimension mismatch")
            if not np.isfinite(vectors).all():
                raise ValueError("Nonfinite child embedding")
            self.index.add(vectors)
        if self.index.ntotal != len(self.windows):
            raise ValueError("Child vector count mismatch")

    def search(self, query, k=30):
        if k < 1:
            return []
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(q, self.index.ntotal)
        best = {}
        for child_id, score in zip(ids[0], scores[0]):
            if child_id < 0:
                continue
            parent = int(self.parent_ids[child_id])
            best.setdefault(parent, (int(child_id), float(score)))
        pool = max(100, 2*k)
        dense = unique_parent_ranking(ids[0], self.parent_ids, pool)
        lexical_scores = self.bm25.scores(query)
        lexical_children = [i for i in np.argsort(-lexical_scores, kind="stable") if lexical_scores[i] > 0]
        lexical = unique_parent_ranking(lexical_children, self.parent_ids, pool)
        parents = reciprocal_rank_fusion([dense, lexical])[:k]
        results = []
        for parent in parents:
            child_id, score = best[parent]
            record = self.records[parent]
            meta = {key: value for key, value in record.items() if key != "text"}
            meta.update({
                "retrieval_granularity": "child_window",
                "matched_window": self.windows[child_id],
                "parent_cosine": float(self.parent_index.reconstruct(parent) @ q[0]),
            })
            results.append(Evidence(
                chunk_id=str(record["chunk_id"]),
                text=record["text"],
                source_id=str(record.get("source_id", "")),
                title=record.get("document_title", ""),
                url=record.get("source_url", ""),
                score=score, metadata=meta,
            ))
        return results
