import re

def rewrite_query(question:str)->str:
    q=re.sub(r"\s+"," ",question.strip())
    replacements={"sugar":"blood glucose","a1c":"HbA1c"}
    for a,b in replacements.items(): q=re.sub(rf"\b{re.escape(a)}\b",b,q,flags=re.I)
    return q
