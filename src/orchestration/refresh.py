from pathlib import Path
import json, subprocess
from datetime import datetime, timezone

def knowledge_refresh_due(manifest_path:str|Path, max_age_days:int=30)->bool:
    p=Path(manifest_path)
    if not p.exists(): return True
    m=json.loads(p.read_text())
    created=datetime.fromisoformat(m["created_at"]); now=datetime.now(timezone.utc)
    return (now-created).days >= max_age_days

def rebuild_index(command=None):
    command=command or ["python","scripts/rebuild_index.py"]
    return subprocess.run(command,check=True,capture_output=True,text=True).stdout
