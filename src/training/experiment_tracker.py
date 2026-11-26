from pathlib import Path
import json
from datetime import datetime, timezone

class ExperimentTracker:
    def __init__(self,path="artifacts/experiments.jsonl"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self, run_name:str, params:dict, metrics:dict, artifacts:dict|None=None):
        row={"run_name":run_name,"created_at":datetime.now(timezone.utc).isoformat(),"params":params,"metrics":metrics,"artifacts":artifacts or {}}
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        return row
