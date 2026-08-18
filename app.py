import streamlit as st, joblib, pandas as pd
model=joblib.load("src/neonatal_risk_model.joblib")
st.title("Neonatal Health Risk Prediction using XAI")
st.write("Educational demonstration only; not a clinical decision system.")
