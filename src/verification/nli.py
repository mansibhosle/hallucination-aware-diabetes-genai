"""Experimental sentence-level NLI grounding checks, not medical fact checking."""
import re
import numpy as np

def sentences(answer):
    clean=re.sub(r"\[S\d+\]", "", answer, flags=re.I).strip()
    return [s.strip(" \t-*") for s in re.split(r"(?<=[.!?])\s+|\n+",clean) if s.strip(" \t-*")]

def classify_pairs(claims, evidence, probs, entail_min=.80, contradiction_min=.80):
    # probs: [claim, passage, (contradiction, entailment, neutral)]
    probs=np.asarray(probs,dtype=float)
    if probs.shape != (len(claims),len(evidence),3):
        raise ValueError("Invalid NLI score shape")
    if not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs>1):
        raise ValueError("Invalid NLI scores")
    details=[]
    for i,claim in enumerate(claims):
        e=probs[i,:,1]; c=probs[i,:,0]
        best=int(np.argmax(e))
        if float(c.max())>=contradiction_min:
            status="contradicted"
        elif float(e[best])>=entail_min:
            status="supported"
        else:
            status="insufficient"
        details.append({"status":status,"chunk_id":evidence[best].chunk_id,
                        "entailment":float(e[best]),"max_contradiction":float(c.max())})
    unsupported=[claim for claim,d in zip(claims,details) if d["status"]!="supported"]
    return {"support":sum(d["status"]=="supported" for d in details)/max(1,len(claims)),
            "unsupported_claims":unsupported,"details":details}

class NLIVerifier:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small", device="cpu", entail_min=.80, contradiction_min=.80):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch=torch
        self.device=device
        if not (0 < entail_min <= 1 and 0 < contradiction_min <= 1):
            raise ValueError("NLI thresholds must be in (0,1]")
        self.entail_min=entail_min
        self.contradiction_min=contradiction_min
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
        expected=["contradiction","entailment","neutral"]
        if set(labels)!=set(expected):
            raise ValueError(f"Unknown NLI label mapping: {labels}")
        self.order=[labels[x] for x in expected]

    def __call__(self, answer, evidence):
        claims=sentences(answer)
        if not claims or not evidence:
            return {"support":0.0,"unsupported_claims":[answer or "empty answer"],"details":[]}
        premises=[e.text for _ in claims for e in evidence]
        hypotheses=[claim for claim in claims for _ in evidence]
        batches=[]
        with self.torch.inference_mode():
            for start in range(0,len(premises),8):
                inputs=self.tokenizer(premises[start:start+8],hypotheses[start:start+8],
                    padding=True,truncation="only_first",max_length=512,return_tensors="pt").to(self.device)
                logits=self.model(**inputs).logits
                batches.append(self.torch.softmax(logits,dim=-1)[:,self.order].cpu().numpy())
        probs=np.concatenate(batches).reshape(len(claims),len(evidence),3)
        return classify_pairs(claims,evidence,probs,self.entail_min,self.contradiction_min)
