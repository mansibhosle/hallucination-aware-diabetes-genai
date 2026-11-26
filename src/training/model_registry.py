from pathlib import Path
import json
from datetime import datetime, timezone

class ModelRegistry:
    def __init__(self,path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self):
        if not self.path.exists(): return {"models":[],"production":None}
        return json.loads(self.path.read_text(encoding="utf-8"))
    def register(self, model_id:str, base_model:str, adapter_path:str, dataset_version:str, metrics:dict, params:dict):
        d=self._load(); d["models"].append({"model_id":model_id,"base_model":base_model,"adapter_path":adapter_path,"dataset_version":dataset_version,"metrics":metrics,"params":params,"registered_at":datetime.now(timezone.utc).isoformat()}); self.path.write_text(json.dumps(d,indent=2),encoding="utf-8")
    def promote(self,model_id:str):
        d=self._load(); assert any(m["model_id"]==model_id for m in d["models"]), "unknown model"; d["production"]=model_id; self.path.write_text(json.dumps(d,indent=2),encoding="utf-8")
    def rollback(self,previous_model_id:str): self.promote(previous_model_id)
