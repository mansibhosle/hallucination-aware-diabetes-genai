from ..schemas import Evidence

class Reranker:
    def __init__(self, backend="heuristic", model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.backend=backend; self.model_name=model_name; self.model=None
        if backend=="cross_encoder":
            from sentence_transformers import CrossEncoder
            self.model=CrossEncoder(model_name, device=device)

    def rerank(self, question: str, evidence: list[Evidence], k: int=5) -> list[Evidence]:
        if not evidence: return []
        if self.backend=="none":
            return evidence[:k]
        if self.model is not None:
            scores=self.model.predict([(question,e.metadata.get("matched_window",e.text)) for e in evidence])
            for e,s in zip(evidence,scores): e.rerank_score=float(s)
        else:
            for e in evidence: e.rerank_score=e.score
        return sorted(evidence, key=lambda x:x.rerank_score, reverse=True)[:k]
