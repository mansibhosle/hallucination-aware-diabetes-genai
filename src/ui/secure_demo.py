"""Restricted sample-only portfolio demo; no private model or corpus."""
import hmac
import os
import sys
from pathlib import Path
import tempfile
import streamlit as st

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
st.set_page_config(page_title="Diabetes GenAI Research Demo")
st.title("Diabetes GenAI — portfolio demonstration")
st.caption("CPU sample-data mode, not trained Phi-3 inference.")
st.warning("Research only. Not medical advice. No real patient data or free-text questions.")

password=os.environ.get("DEMO_PASSWORD","")
if len(password)<16:
    st.error("Configure a demo password of at least 16 characters on the host.")
    st.stop()
if not st.session_state.get("authorized",False):
    with st.form("login"):
        entered=st.text_input("Demo access password",type="password")
        submitted=st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(entered.encode(),password.encode()):
            st.session_state.authorized=True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
if st.sidebar.button("Sign out"):
    st.session_state.clear()
    st.rerun()

@st.cache_resource
def load_demo():
    from src.config import Settings
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.generation.generator import GroundedGenerator
    from src.pipeline import Platform
    settings=Settings(generation_backend="extractive",embedding_backend="tfidf",
        verification_backend="lexical",db_path=Path(tempfile.mkdtemp(prefix="diabetes-demo-"))/"demo.sqlite",
        top_k_candidates=5,top_k_final=5,retrieval_gate=.60,tfidf_retrieval_gate=.15,
        base_confidence_threshold=.68)
    index=ROOT/"artifacts/index/hybrid.joblib"
    if not index.is_file():
        raise FileNotFoundError("Build the sample index first")
    return Platform(HybridRetriever.load(index),Reranker("heuristic"),GroundedGenerator("extractive"),settings)
cases=[
    "What is prediabetes and which laboratory measures are used to diagnose it, including A1C and fasting plasma glucose?",
    "Why is foot care important in diabetes?",
    "How many units of insulin should I take?",
    "Which stock should I buy?",
]
question=st.selectbox("Choose a scripted test case",cases)
if st.button("Run example"):
    try:
        result=load_demo().answer(question)
    except Exception:
        st.error("Demo unavailable. Contact the host.")
        st.stop()
    st.subheader(result["decision"])
    if result["decision"]=="ANSWER": st.success(result["answer"])
    else: st.warning(result["answer"])
    st.metric("Heuristic confidence (not probability)",f"{result['confidence']:.3f}")
    with st.expander("Sample evidence and trace"):
        for e in result["evidence"]:
            st.write(e["title"])
            st.write(e["text"])
        st.code(result["trace_id"])
