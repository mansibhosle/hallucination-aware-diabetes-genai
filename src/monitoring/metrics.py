from collections import Counter
import sqlite3, json

def snapshot(db_path: str) -> dict:
    con=sqlite3.connect(db_path); rows=con.execute("SELECT decision, confidence, risk_class FROM interactions").fetchall(); con.close()
    if not rows: return {"count":0}
    decisions=Counter(r[0] for r in rows); risks=Counter(r[2] for r in rows)
    return {"count":len(rows),"answer_rate":decisions.get("ANSWER",0)/len(rows),"abstain_rate":1-decisions.get("ANSWER",0)/len(rows),"avg_confidence":sum(r[1] or 0 for r in rows)/len(rows),"decisions":dict(decisions),"risk_classes":dict(risks)}
