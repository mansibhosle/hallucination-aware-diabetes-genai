from __future__ import annotations
from pathlib import Path
import json, math, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..schemas import Evidence

class HybridRetriever:
    """CPU-friendly hybrid retriever. Swap semantic arm to SentenceTransformer/FAISS in production."""
    def __init__(self, semantic_weight: float=.7, lexical_weight: float=.3):
        self.semantic_weight=semantic_weight; self.lexical_weight=lexical_weight
        self.chunks=[]; self.vectorizer=None; self.matrix=None

    def fit(self, chunks: list[dict]):
        self.chunks=chunks
        corpus=[c["text"] for c in chunks]
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=50000)
        self.matrix=self.vectorizer.fit_transform(corpus)
        return self

    def save(self, path: str | Path):
        import joblib
        p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks":self.chunks,"vectorizer":self.vectorizer,"matrix":self.matrix}, p)

    @classmethod
    def load(cls, path: str | Path):
        import joblib
        d=joblib.load(path); obj=cls(); obj.chunks=d["chunks"]; obj.vectorizer=d["vectorizer"]; obj.matrix=d["matrix"]; return obj

    def search(self, query: str, k: int=20) -> list[Evidence]:
        if self.vectorizer is None or self.matrix is None: return []
        q=self.vectorizer.transform([query])
        semantic=cosine_similarity(q,self.matrix).ravel()
        # lexical exact-term overlap complements TF-IDF cosine
        qterms=set(re.findall(r"[a-z0-9]+", query.lower()))
        lexical=np.array([len(qterms & set(re.findall(r"[a-z0-9]+", c["text"].lower())))/max(len(qterms),1) for c in self.chunks])
        score=self.semantic_weight*semantic + self.lexical_weight*lexical
        idx=np.argsort(score)[::-1][:k]
        return [Evidence(chunk_id=self.chunks[i]["chunk_id"], text=self.chunks[i]["text"], source_id=self.chunks[i].get("source_id",""), title=self.chunks[i].get("title",""), url=self.chunks[i].get("url",""), score=float(score[i]), metadata={k:v for k,v in self.chunks[i].items() if k not in {"text"}}) for i in idx]
