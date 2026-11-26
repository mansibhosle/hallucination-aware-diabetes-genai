from fastapi import FastAPI, HTTPException
from ..schemas import AnswerRequest, FeedbackRequest
from ..pipeline import Platform
from ..config import SETTINGS
from ..storage.sqlite import log_feedback
from ..monitoring.metrics import snapshot

app=FastAPI(title="Diabetes GenAI Reliability Platform",version="1.0.0")
_system=None

def system():
    global _system
    if _system is None: _system=Platform.load_default()
    return _system

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/v1/answer")
def answer(req:AnswerRequest):
    try: return system().answer(req.question,req.user_role)
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/v1/feedback")
def feedback(req:FeedbackRequest):
    log_feedback(SETTINGS.db_path,req.trace_id,req.rating,req.comment); return {"saved":True}

@app.get("/v1/monitoring")
def monitoring(): return snapshot(SETTINGS.db_path)
