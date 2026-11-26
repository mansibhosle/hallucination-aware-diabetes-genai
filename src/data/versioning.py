from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

def build_manifest(chunks: list[dict], output: str | Path) -> dict:
    canonical="\n".join(sorted(c.get("sha256", "") for c in chunks))
    version=hashlib.sha256(canonical.encode()).hexdigest()[:12]
    manifest={
        "kb_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "source_ids": sorted({c.get("source_id","") for c in chunks}),
    }
    p=Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
