from pathlib import Path
import yaml, hashlib, json

class SourceRegistry:
    def __init__(self, config_path: str | Path = "configs/sources.yaml"):
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.sources = {x["source_id"]: x for x in raw.get("approved_sources", [])}

    def get(self, source_id: str) -> dict:
        if source_id not in self.sources:
            raise ValueError(f"Unapproved source_id: {source_id}")
        return self.sources[source_id]

    def assert_ingest_allowed(self, source_id: str) -> dict:
        item = self.get(source_id)
        if item.get("status") != "approved" or not item.get("redistributable", False):
            raise PermissionError(f"Source {source_id} is not approved for local full-text ingestion")
        return item

def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def write_audit(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
