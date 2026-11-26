from dataclasses import dataclass, asdict
import re

@dataclass
class QualityResult:
    accepted: bool
    issues: list[str]
    word_count: int

    def to_dict(self):
        return asdict(self)

def assess_text(text: str, min_words: int = 40) -> QualityResult:
    issues=[]
    words=text.split()
    if len(words) < min_words: issues.append("too_short")
    if not text.strip(): issues.append("empty")
    if len(re.findall(r"\ufffd", text)) > 2: issues.append("encoding_artifacts")
    alpha = sum(c.isalpha() for c in text)
    if text and alpha/max(len(text),1) < 0.45: issues.append("low_alpha_ratio")
    return QualityResult(not issues, issues, len(words))
