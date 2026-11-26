def classification_metrics(rows:list[dict])->dict:
    n=len(rows) or 1
    correct=sum(bool(r.get("correct")) for r in rows)
    should=[r for r in rows if r.get("should_abstain")]
    abstained=[r for r in rows if str(r.get("decision","")).startswith("ABSTAIN")]
    true_abst=sum(1 for r in should if str(r.get("decision","")).startswith("ABSTAIN"))
    false_abst=sum(1 for r in rows if not r.get("should_abstain") and str(r.get("decision","")).startswith("ABSTAIN"))
    prec=true_abst/max(true_abst+false_abst,1); rec=true_abst/max(len(should),1); f1=2*prec*rec/max(prec+rec,1e-9)
    answered=[r for r in rows if r.get("decision")=="ANSWER"]
    halluc=sum(bool(r.get("unsupported")) for r in answered)/max(len(answered),1)
    return {"end_to_end_accuracy":correct/n,"abstention_precision":prec,"abstention_recall":rec,"abstention_f1":f1,"coverage":len(answered)/n,"evidence_based_hallucination":halluc}
