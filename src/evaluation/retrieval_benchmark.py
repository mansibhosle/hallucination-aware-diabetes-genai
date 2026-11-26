"""Source-level retrieval metrics for the original validation split.

Source labels are proxies for relevance, not passage-level clinical judgments.
"""
import ast
import re
import pandas as pd
import numpy as np

REQUIRED = {"question_id", "question", "category", "should_abstain", "reference_source_ids", "split"}

def parse_source_ids(value):
    value = str(value).strip()
    if not value:
        return set()
    if value.startswith(("[", "(")):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)) or not all(isinstance(x,str) for x in parsed):
            raise ValueError("Expected a list of source IDs")
        return {x.strip() for x in parsed if x.strip()}
    return {x.strip("\"'") for x in re.split(r"[,;|\s]+",value) if x.strip("\"'")}

def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true","1","1.0","yes"}:
        return True
    if value in {"false","0","0.0","no"}:
        return False
    raise ValueError(f"Unexpected should_abstain value: {value!r}")

def load_validation(path, known_sources):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED-set(frame.columns))}")
    validation = frame.loc[frame["split"].str.strip().str.lower().eq("validation")]
    if validation.empty:
        raise ValueError("No original validation rows; no random split will be created")
    if validation["question_id"].duplicated().any():
        raise ValueError("Duplicate validation question IDs")
    cases, excluded = [], []
    for row in validation.to_dict("records"):
        if parse_bool(row["should_abstain"]):
            excluded.append({"question_id":row["question_id"],"reason":"should_abstain"})
            continue
        labels = parse_source_ids(row["reference_source_ids"])
        unknown = labels-known_sources
        if not labels or unknown:
            excluded.append({"question_id":row["question_id"],"reason":"missing_or_unmapped_source_ids","unknown_ids":sorted(unknown)})
            continue
        if not row["question"].strip() or not row["question_id"].strip():
            raise ValueError("Empty validation question or question ID")
        cases.append({"question_id":row["question_id"],"question":row["question"],
                      "category":row["category"],"reference_source_ids":labels})
    return cases, excluded, len(validation)

def source_metrics(reference_ids, ranked_sources, k=5):
    if not reference_ids or k<1:
        raise ValueError("Nonempty reference IDs and positive k required")
    top=list(ranked_sources)[:k]
    first=next((i for i,source in enumerate(top,1) if source in reference_ids),None)
    return {"hit_at_1":float(bool(top) and top[0] in reference_ids),
            "hit_at_5":float(first is not None),
            "mrr_at_5":1.0/first if first else 0.0,
            "source_recall_at_5":len(set(top)&reference_ids)/len(reference_ids)}

def aggregate(rows):
    if not rows:
        raise ValueError("Cannot aggregate an empty benchmark")
    metrics=["hit_at_1","hit_at_5","mrr_at_5","source_recall_at_5","candidate_source_hit"]
    return {"n":len(rows),**{m:float(np.mean([r[m] for r in rows])) for m in metrics},
            "mean_latency_ms":float(np.mean([r["latency_ms"] for r in rows])),
            "p95_latency_ms":float(np.percentile([r["latency_ms"] for r in rows],95))}
