from pathlib import Path
import sqlite3, json
from datetime import datetime, timezone

def connect(path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    con=sqlite3.connect(p)
    con.execute("""CREATE TABLE IF NOT EXISTS interactions(trace_id TEXT PRIMARY KEY, created_at TEXT, question TEXT, decision TEXT, reason TEXT, confidence REAL, risk_class TEXT, payload TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, created_at TEXT, rating TEXT, comment TEXT)""")
    con.commit(); return con

def log_interaction(path, payload: dict):
    con=connect(path)
    con.execute("INSERT OR REPLACE INTO interactions VALUES(?,?,?,?,?,?,?,?)", (payload["trace_id"],datetime.now(timezone.utc).isoformat(),payload["question"],payload["decision"],payload.get("reason",""),payload.get("confidence",0.0),payload.get("risk_class",""),json.dumps(payload,ensure_ascii=False)))
    con.commit(); con.close()

def log_feedback(path, trace_id, rating, comment=""):
    con=connect(path); con.execute("INSERT INTO feedback(trace_id,created_at,rating,comment) VALUES(?,?,?,?)",(trace_id,datetime.now(timezone.utc).isoformat(),rating,comment)); con.commit(); con.close()
