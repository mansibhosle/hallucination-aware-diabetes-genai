import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.final_metrics import reviewed_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--review",type=Path,required=True)
    a=p.parse_args()
    original=pd.read_csv(a.run_dir/"review.csv",dtype=str,keep_default_na=False)
    reviewed=pd.read_csv(a.review,dtype=str,keep_default_na=False)
    manifest=json.loads((a.run_dir/"manifest.json").read_text())
    if len(original)!=manifest["n_expected"] or len(reviewed)!=len(original):
        raise ValueError("Incomplete evaluation")
    cols=[c for c in original.columns if c not in {"factual_correct","evidence_supported","safe_under_review_rubric","review_notes"}]
    if not original[cols].equals(reviewed[cols]):
        raise ValueError("Review changed original inference fields or row order")
    metrics=reviewed_metrics(reviewed)
    (a.run_dir/"reviewed_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
