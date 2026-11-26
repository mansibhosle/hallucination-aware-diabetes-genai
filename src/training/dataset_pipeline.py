from pathlib import Path
import json, hashlib

def build_sft_dataset(candidates:list[dict], output:str|Path) -> dict:
    approved=[x for x in candidates if x.get("approved") is True]
    canonical=json.dumps(approved, sort_keys=True, ensure_ascii=False)
    version=hashlib.sha256(canonical.encode()).hexdigest()[:12]
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in approved),encoding="utf-8")
    return {"dataset_version":version,"candidate_count":len(candidates),"approved_count":len(approved),"rejected_count":len(candidates)-len(approved)}
