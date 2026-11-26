import streamlit as st
import requests

st.set_page_config(page_title="GenAI Operations Dashboard",layout="wide")
st.title("GenAI Operations Dashboard")
api=st.sidebar.text_input("API URL","http://localhost:8000")
try:
    data=requests.get(f"{api}/v1/monitoring",timeout=15).json()
except Exception as e:
    st.error(str(e)); st.stop()
st.metric("Interactions",data.get("count",0))
c1,c2,c3=st.columns(3)
c1.metric("Answer rate",f"{data.get('answer_rate',0):.1%}")
c2.metric("Abstain rate",f"{data.get('abstain_rate',0):.1%}")
c3.metric("Avg confidence",f"{data.get('avg_confidence',0):.3f}")
st.subheader("Decision distribution"); st.json(data.get("decisions",{}))
st.subheader("Risk classes"); st.json(data.get("risk_classes",{}))
