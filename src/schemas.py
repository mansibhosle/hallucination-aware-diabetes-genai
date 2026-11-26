from typing import Any, Literal
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    chunk_id: str
    text: str
    source_id: str
    title: str = ""
    url: str = ""
    score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

class AnswerRequest(BaseModel):
    question: str
    user_role: Literal["patient", "clinician", "researcher"] = "patient"
    conversation_id: str | None = None

class AnswerResponse(BaseModel):
    question: str
    decision: str
    answer: str
    reason: str = ""
    confidence: float = 0.0
    risk_class: str = "general_education"
    evidence: list[Evidence] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    trace_id: str

class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["helpful", "not_helpful", "incorrect", "unsupported"]
    comment: str = ""
