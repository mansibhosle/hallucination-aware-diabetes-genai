from ..schemas import Evidence

class GroundedGenerator:
    def __init__(self, backend="extractive", model_name="microsoft/Phi-3-mini-4k-instruct", adapter_path=""):
        self.backend=backend; self.model_name=model_name; self.adapter_path=adapter_path
        self.pipe=None
        if backend=="transformers":
            from transformers import pipeline
            self.pipe=pipeline("text-generation", model=model_name, device_map="auto")

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return "Insufficient evidence to answer from the approved knowledge base."
        if self.pipe is None:
            # Safe local fallback for lifecycle testing: return the strongest evidence excerpt.
            return evidence[0].text
        ctx="\n\n".join(f"[S{i+1}] {e.text}" for i,e in enumerate(evidence))
        prompt=("You are an evidence-grounded diabetes knowledge-support assistant. Answer only from the supplied evidence. "
                "Do not provide individualized dosing or medication changes. If evidence is insufficient, say so.\n\n"
                f"Question: {question}\n\nEvidence:\n{ctx}\n\nAnswer:")
        out=self.pipe(prompt, max_new_tokens=120, do_sample=False, return_full_text=False)
        return out[0]["generated_text"].strip()
