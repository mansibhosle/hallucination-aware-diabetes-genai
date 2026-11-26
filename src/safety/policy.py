import re
from dataclasses import dataclass

@dataclass
class PolicyDecision:
    allowed: bool
    intent: str
    risk_class: str
    message: str = ""

EMERGENCY_PATTERNS=[r"\bunconscious\b", r"\bseizure\b", r"\bcan't wake\b", r"\bcannot wake\b", r"\bsevere hypogly", r"\bketoacidosis\b", r"\bdka\b"]
DOSE_PATTERNS=[r"\bhow many units\b", r"\bhow much insulin\b", r"\bdouble my insulin\b", r"\bextra insulin\b", r"\bmore insulin\b", r"\bshould i (?:increase|decrease|double|skip|stop)\b"]
MED_PATTERNS=[r"\bstop my medication\b", r"\bchange my medication\b", r"\bswitch my medication\b", r"\bincrease my medication\b", r"\bdecrease my medication\b"]
DEVICE_PATTERNS=[r"\bchange my pump setting", r"\badjust my pump", r"\bchange my device setting"]
OUT_DOMAIN=[r"\bstock\b",r"\bvisa\b",r"\bweather\b",r"\bfootball\b",r"\brecipe\b"]

def _match(patterns,q): return any(re.search(p,q,re.I) for p in patterns)

def classify(question: str) -> PolicyDecision:
    q=question.strip()
    if _match(EMERGENCY_PATTERNS,q):
        return PolicyDecision(False,"emergency","emergency","This may require urgent medical attention. The prototype cannot manage emergencies. Seek emergency medical care or local emergency services now.")
    if _match(DOSE_PATTERNS,q):
        return PolicyDecision(False,"individualized_insulin_dose","medication_high_risk","I cannot provide an individualized insulin dose. Follow your prescribed plan or contact your healthcare professional.")
    if _match(MED_PATTERNS,q):
        return PolicyDecision(False,"medication_change","medication_high_risk","I cannot recommend starting, stopping, switching, or changing a prescribed medication. Please contact your healthcare professional.")
    if _match(DEVICE_PATTERNS,q):
        return PolicyDecision(False,"device_setting_change","device_high_risk","I cannot recommend patient-specific device-setting changes. Please follow your prescribed settings or contact your healthcare professional.")
    if _match(OUT_DOMAIN,q):
        return PolicyDecision(False,"out_of_domain","out_of_domain","This question is outside the scope of the diabetes knowledge-support system.")
    risk="pregnancy" if re.search(r"pregnan|gestational",q,re.I) else ("diagnosis_general" if re.search(r"diagnos|criteria|a1c|hba1c",q,re.I) else "general_education")
    return PolicyDecision(True,"general_information",risk)
