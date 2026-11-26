import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Evidence Assistant",layout="wide")
st.title("Diabetes Evidence Assistant")
st.caption("Research prototype — evidence-grounded knowledge support with risk-aware abstention")
st.warning("Not a medical device. Do not use for autonomous diagnosis, prescribing, dosing, or emergency decisions.")
api=st.sidebar.text_input("API URL","http://localhost:8000")
role=st.sidebar.selectbox("Mode",["patient","clinician","researcher"])
q=st.text_area("Ask a diabetes-related question",height=100)
if st.button("Submit",type="primary") and q.strip():
    r=requests.post(f"{api}/v1/answer",json={"question":q,"user_role":role},timeout=180)
    data=r.json(); st.subheader(data["decision"])
    if data["decision"]=="ANSWER": st.success(data["answer"])
    else: st.error(data["answer"])
    c1,c2,c3=st.columns(3); c1.metric("Confidence",f"{data.get('confidence',0):.3f}"); c2.metric("Risk class",data.get("risk_class","")); c3.metric("Evidence",len(data.get("evidence",[])))
    with st.expander("Evidence and trace"):
        for e in data.get("evidence",[]): st.write(f"**{e.get('title') or e.get('source_id')}** — score {max(e.get('score',0),e.get('rerank_score',0)):.3f}"); st.write(e.get("text","")); st.divider()
        st.code(data.get("trace_id",""))
    st.write("Was this useful?")
    cols=st.columns(4)
    for i,label in enumerate(["helpful","not_helpful","incorrect","unsupported"]):
        if cols[i].button(label,key=label): requests.post(f"{api}/v1/feedback",json={"trace_id":data["trace_id"],"rating":label,"comment":""},timeout=30); st.toast("Feedback saved")
