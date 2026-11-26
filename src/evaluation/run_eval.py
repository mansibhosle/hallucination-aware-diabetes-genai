import argparse, json, time
from pathlib import Path
from .metrics import classification_metrics
from ..pipeline import Platform

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--file",required=True); args=ap.parse_args()
    system=Platform.load_default()
    rows=[]
    for line in Path(args.file).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        item=json.loads(line); t=time.perf_counter(); res=system.answer(item["question"]); latency=time.perf_counter()-t
        pred_abst=res["decision"].startswith("ABSTAIN")
        correct=(pred_abst==bool(item.get("should_abstain"))) if item.get("should_abstain") else res["decision"]=="ANSWER"
        rows.append({**item,**res,"latency":latency,"correct":correct})
    print(json.dumps({"metrics":classification_metrics(rows),"n":len(rows)},indent=2))
if __name__=="__main__": main()
