from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import json
from .config import SETTINGS
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import Reranker
from .generation.generator import GroundedGenerator
from .verification.claims import verify_claim_support
from .safety.policy import classify
from .confidence.scoring import score as confidence_score, threshold_for
from .storage.sqlite import log_interaction

class Platform:
    def __init__(self,retriever,reranker,generator,settings=SETTINGS):
        self.retriever=retriever; self.reranker=reranker; self.generator=generator; self.settings=settings

    @classmethod
    def load_default(cls):
        if SETTINGS.embedding_backend == "faiss":
            from .retrieval.faiss_retriever import FaissRetriever
            r=FaissRetriever(SETTINGS.faiss_index_path, SETTINGS.faiss_metadata_path,
                SETTINGS.faiss_build_info_path, SETTINGS.guideline_chunks_path,
                SETTINGS.embedding_model, SETTINGS.embedding_device, SETTINGS.faiss_search_mode)
        elif SETTINGS.embedding_backend == "tfidf":
            index=SETTINGS.index_dir/"hybrid.joblib"
            if not index.exists():
                raise FileNotFoundError("Index missing. Run: python scripts/bootstrap_demo.py")
            r=HybridRetriever.load(index)
        else:
            raise ValueError("Unknown embedding backend")
        rr=Reranker(SETTINGS.rerank_backend, SETTINGS.rerank_model, SETTINGS.rerank_device)
        g=GroundedGenerator(SETTINGS.generation_backend, SETTINGS.model_name, SETTINGS.model_adapter_path, SETTINGS.max_new_tokens)
        return cls(r,rr,g)

    def answer(self, question:str, user_role:str="patient") -> dict:
        trace_id=str(uuid4())
        policy=classify(question)
        if not policy.allowed:
            payload={"question":question,"decision":f"ABSTAIN_{policy.intent.upper()}","answer":policy.message,"reason":policy.intent,"confidence":0.0,"risk_class":policy.risk_class,"evidence":[],"metrics":{},"trace_id":trace_id}
            log_interaction(self.settings.db_path,payload); return payload

        candidates=self.retriever.search(question,self.settings.top_k_candidates)
        evidence=self.reranker.rerank(question,candidates,self.settings.top_k_final)
        retrieval=max([max(0.0,min(1.0,e.score)) for e in evidence],default=0.0)
        gate = getattr(self.settings, "tfidf_retrieval_gate", self.settings.retrieval_gate) if getattr(self.settings, "embedding_backend", "tfidf") == "tfidf" else self.settings.retrieval_gate
        availability=1.0 if evidence and retrieval>=gate else 0.0
        if availability==0:
            payload={"question":question,"decision":"ABSTAIN_EVIDENCE","answer":"The approved knowledge base does not contain sufficiently strong evidence to answer this question.","reason":"insufficient_evidence","confidence":0.45*retrieval,"risk_class":policy.risk_class,"evidence":[e.model_dump() for e in evidence],"metrics":{"retrieval_quality":retrieval,"evidence_availability":availability},"trace_id":trace_id}
            log_interaction(self.settings.db_path,payload); return payload

        answer=self.generator.generate(question,evidence)
        if answer.startswith("ABSTAIN: Generation reached its token budget"):
            payload={"question":question,"decision":"ABSTAIN_TRUNCATED_GENERATION","answer":"The model could not complete its response within the generation budget.","reason":"generation_token_limit","confidence":0.0,"risk_class":policy.risk_class,"evidence":[e.model_dump() for e in evidence],"metrics":{"retrieval_quality":retrieval,"evidence_availability":availability},"trace_id":trace_id}
            log_interaction(self.settings.db_path,payload); return payload
        if not answer.strip() or answer.strip().lower().startswith(("abstain", "insufficient evidence", "the evidence does not support", "the available evidence does not support", "i cannot", "i can't")):
            payload={"question":question,"decision":"ABSTAIN_MODEL_REFUSAL","answer":"The model did not provide a supported answer.","reason":"model_refusal_or_empty","confidence":0.0,"risk_class":policy.risk_class,"evidence":[e.model_dump() for e in evidence],"metrics":{"retrieval_quality":retrieval,"evidence_availability":availability},"trace_id":trace_id}
            log_interaction(self.settings.db_path,payload); return payload
        verification=verify_claim_support(answer,evidence)
        support=verification["support"]
        conf=confidence_score(retrieval,availability,support)
        threshold=threshold_for(policy.risk_class,self.settings.base_confidence_threshold)
        if conf < threshold or verification["unsupported_claims"]:
            decision="ABSTAIN_CONFIDENCE"; final="The available evidence was not strong enough to support a reliable answer."
            reason="low_confidence_or_unsupported_claim"
        else:
            decision="ANSWER"; final=answer; reason=""
        payload={"question":question,"decision":decision,"answer":final,"reason":reason,"confidence":conf,"risk_class":policy.risk_class,"evidence":[e.model_dump() for e in evidence],"metrics":{"retrieval_quality":retrieval,"evidence_availability":availability,"answer_evidence_support":support,"threshold":threshold},"trace_id":trace_id}
        log_interaction(self.settings.db_path,payload); return payload
