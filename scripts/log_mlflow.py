"""Local experiment tracking; no raw question or evidence text."""
import argparse,json
from pathlib import Path
import pandas as pd
import mlflow

def log_metrics(values):
    mlflow.log_metrics({k:float(v) for k,v in values.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark-dir",type=Path,default=Path("outputs/retrieval_benchmark"))
    p.add_argument("--run-dir",type=Path)
    a=p.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
    mlflow.set_experiment("diabetes-genai-retrieval")
    bm=json.loads((a.benchmark_dir/"manifest.json").read_text())
    summary=pd.read_csv(a.benchmark_dir/"summary.csv")
    if len(summary)!=4: raise ValueError("Incomplete benchmark")
    for row in summary.to_dict("records"):
        with mlflow.start_run(run_name=row["configuration"]):
            mlflow.log_params({"configuration":row["configuration"],"split":"validation",
                **{k:bm[k] for k in ("candidate_count","embedding_model","reranker_model","evaluation_sha256","index_sha256","git_commit")}})
            log_metrics(row)
    if a.run_dir:
        mlflow.set_experiment("diabetes-genai-end-to-end")
        m=json.loads((a.run_dir/"manifest.json").read_text())
        reviewed=(a.run_dir/"reviewed_metrics.json").exists()
        name="reviewed_metrics.json" if reviewed else "automatic_metrics.json"
        metrics=json.loads((a.run_dir/name).read_text())
        with mlflow.start_run(run_name=m["configuration"]+"-"+m["split"]):
            mlflow.log_params({k:m[k] for k in ("configuration","split","evaluation_sha256","index_sha256","adapter_sha256","git_commit")})
            mlflow.log_params({k:m["settings"][k] for k in ("verification_backend","retrieval_gate","base_confidence_threshold","nli_entailment_threshold","nli_contradiction_threshold","max_new_tokens")})
            mlflow.set_tag("review_status","manual_review_complete" if reviewed else "decision_metrics_only")
            log_metrics(metrics)
    print("Aggregate experiments logged locally: outputs/mlflow.db")
if __name__=="__main__": main()
