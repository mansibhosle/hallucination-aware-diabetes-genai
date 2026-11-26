import re

PATTERNS={
    "email":r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone":r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "ssn":r"\b\d{3}-\d{2}-\d{4}\b",
}

def detect_phi(text:str)->list[str]:
    return [name for name,p in PATTERNS.items() if re.search(p,text,re.I)]

def redact_phi(text:str)->str:
    out=text
    for name,p in PATTERNS.items(): out=re.sub(p,f"[REDACTED_{name.upper()}]",out,flags=re.I)
    return out
