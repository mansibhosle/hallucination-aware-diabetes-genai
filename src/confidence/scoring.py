RISK_THRESHOLDS={"general_education":0.68,"diagnosis_general":0.74,"pregnancy":0.80,"medication_general":0.82}

def score(retrieval_quality: float, evidence_availability: float, answer_evidence_support: float) -> float:
    return 0.45*retrieval_quality + 0.20*evidence_availability + 0.35*answer_evidence_support

def threshold_for(risk_class: str, base: float=.68) -> float:
    return RISK_THRESHOLDS.get(risk_class, base)
