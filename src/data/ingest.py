from pathlib import Path
import json, re, unicodedata
from datetime import datetime, timezone
from .quality import assess_text
from ..governance.registry import SourceRegistry, content_sha256


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_words(text: str, size: int = 180, overlap: int = 35):
    words = text.split()
    step = max(size-overlap, 1)
    for i in range(0, len(words), step):
        chunk = words[i:i+size]
        if len(chunk) < 30: break
        yield " ".join(chunk)


def ingest_text_file(path: str | Path, source_id: str, title: str, url: str = "", registry_path: str = "configs/sources.yaml") -> dict:
    registry = SourceRegistry(registry_path)
    source = registry.assert_ingest_allowed(source_id)
    path=Path(path)
    text=clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    q=assess_text(text)
    if not q.accepted:
        return {"accepted": False, "quality": q.to_dict(), "chunks": []}
    now=datetime.now(timezone.utc).isoformat()
    chunks=[]
    seen=set()
    for n, chunk in enumerate(chunk_words(text)):
        fp=content_sha256(chunk)
        if fp in seen: continue
        seen.add(fp)
        chunks.append({
            "chunk_id": f"{source_id}-{fp[:12]}", "source_id": source_id,
            "organization": source["organization"], "title": title, "url": url,
            "retrieved_at": now, "text": chunk, "sha256": fp, "chunk_number": n
        })
    return {"accepted": True, "quality": q.to_dict(), "document_sha256": content_sha256(text), "chunks": chunks}


def save_ingestion(result: dict, out_path: str | Path) -> None:
    out=Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in result.get("chunks", []): f.write(json.dumps(row, ensure_ascii=False)+"\n")
