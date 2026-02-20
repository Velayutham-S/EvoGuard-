import streamlit as st
from monitor import run_monitor

st.title("🛡️ EvoGuard++ Monitoring Dashboard")

drift_score, severity, action, new_auc = run_monitor()

st.header("📊 Drift Analysis")
st.metric("Global Drift Score", round(drift_score, 3))
st.write("Severity:", severity)

st.header("⚙️ System Action")
st.write(action)

if new_auc:
    st.header("🔄 Model Update")
    st.success(f"New Model AUC: {round(new_auc, 3)}")