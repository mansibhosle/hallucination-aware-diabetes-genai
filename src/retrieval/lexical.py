"""Lexical BM25 ranking and conservative bibliography filtering for research retrieval."""
import math
import re
from collections import Counter
import numpy as np

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def is_reference_chunk(record):
    section = str(record.get("clinical_section", "")).strip().lower()
    text = record.get("text", "").strip()
    if section in {"references", "bibliography", "literature cited", "reference list"}:
        return True
    if re.match(r"^(references|bibliography|literature cited)\b", text, re.I):
        return True
    # Multiple journal-style year/volume/page references suggest a bibliography.
    return len(re.findall(r"\b(?:19|20)\d{2}\s*;\s*\d+(?:\(\d+\))?\s*:\s*\d+", text)) >= 3

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.counts = [Counter(tokens(d)) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=float)
        self.avgdl = max(float(self.lengths.mean()), 1.0)
        df = Counter(t for c in self.counts for t in c)
        n = len(self.counts)
        self.idf = {t: math.log(1+(n-f+0.5)/(f+0.5)) for t,f in df.items()}
        self.k1=k1
        self.b=b

    def scores(self, query):
        score = np.zeros(len(self.counts), dtype=float)
        norm = self.k1*(1-self.b+self.b*self.lengths/self.avgdl)
        for t in set(tokens(query)):
            if t not in self.idf:
                continue
            tf = np.array([c.get(t,0) for c in self.counts], dtype=float)
            score += self.idf[t]*tf*(self.k1+1)/(tf+norm)
        return score

def reciprocal_rank_fusion(rankings, k=60):
    fused = Counter()
    for ranking in rankings:
        for i, idx in enumerate(ranking, 1):
            fused[int(idx)] += 1/(k+i)
    return sorted(fused, key=lambda idx: (-fused[idx],idx))
