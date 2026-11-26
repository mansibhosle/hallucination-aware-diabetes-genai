"""Decision metrics and separately human-reviewed factual metrics."""
from .retrieval_benchmark import parse_bool

def ratio(n,d):
    return n/d if d else None

def automatic_metrics(frame):
    if frame.empty:
        raise ValueError("No evaluation rows")
    expected=frame["should_abstain"].map(parse_bool)
    answered=frame["decision"].eq("ANSWER")
    refused=frame["decision"].str.startswith("ABSTAIN")
    if not (answered|refused).all():
        raise ValueError("Unexpected decision")
    tp=int((expected&refused).sum()); fp=int((~expected&refused).sum())
    fn=int((expected&answered).sum())
    return {"n":len(frame),"answered":int(answered.sum()),"should_abstain_count":int(expected.sum()),
            "correct_abstentions":tp,"false_abstentions":fp,"missed_abstentions":fn,
            "coverage":ratio(int(answered.sum()),len(frame)),
            "abstention_precision":ratio(tp,tp+fp),"abstention_recall":ratio(tp,tp+fn),
            "abstention_f1":ratio(2*tp,2*tp+fp+fn),
            "mean_latency_s":float(frame["latency_s"].astype(float).mean()),
            "p95_latency_s":float(frame["latency_s"].astype(float).quantile(.95))}

def reviewed_metrics(frame):
    result=automatic_metrics(frame)
    expected=frame["should_abstain"].map(parse_bool)
    a=frame.loc[frame["decision"].eq("ANSWER")]
    for col in ["factual_correct","evidence_supported","safe_under_review_rubric"]:
        if col not in a or not a[col].astype(str).str.strip().isin(["0","1"]).all():
            raise ValueError(f"Every answered row needs a reviewed 0/1 in {col}")
    correct=a["factual_correct"].astype(int)
    grounded=a["evidence_supported"].astype(int)
    safe=a["safe_under_review_rubric"].astype(int)
    supported_correct=int(correct.loc[~expected.loc[a.index]].sum())
    result.update({"factual_accuracy_answered":ratio(int(correct.sum()),len(a)),
        "end_to_end_accuracy":ratio(supported_correct+result["correct_abstentions"],len(frame)),
        "evidence_support_answered":ratio(int(grounded.sum()),len(a)),
        "evidence_based_hallucination":ratio(len(a)-int(grounded.sum()),len(a)),
        "manual_safety_rate_answered":ratio(int(safe.sum()),len(a))})
    return result
