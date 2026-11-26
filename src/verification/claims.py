import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..schemas import Evidence

def split_claims(answer: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if len(x.split())>=4]

def verify_claim_support(answer: str, evidence: list[Evidence]) -> dict:
    claims=split_claims(answer)
    if not claims or not evidence:
        return {"support":0.0,"unsupported_claims":claims}
    ev=[e.text for e in evidence]
    vec=TfidfVectorizer(stop_words="english").fit(ev+claims)
    evec=vec.transform(ev); cvec=vec.transform(claims)
    sims=cosine_similarity(cvec,evec)
    best=sims.max(axis=1)
    supported=[b>=0.18 for b in best]
    return {"support": float(sum(supported)/len(supported)), "unsupported_claims":[c for c,s in zip(claims,supported) if not s], "claim_scores":[float(x) for x in best]}
