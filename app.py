# app.py
import os
import json
import time
import logging
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
COLUMNS_PATH = os.getenv("COLUMNS_PATH", "artifacts/columns.json")
APP_ENV = os.getenv("APP_ENV", "dev")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, "r") as f:
        cols = json.load(f)["features"]
    return model, cols

model, FEATURES = load_model()

# Keep simple in-memory metrics (resets on container restart)
latencies_ms = deque(maxlen=500)
error_count = 0
request_count = 0

st.title("Titanic Survival Predictor 🚢")
st.caption(f"Environment: `{APP_ENV}` | Model: `{Path(MODEL_PATH).name}`")

# Basic input UI (tweak defaults as you like)
col1, col2, col3 = st.columns(3)
with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Siblings/Spouses (SibSp)", min_value=0, max_value=8, value=0)
with col2:
    parch = st.number_input("Parents/Children (Parch)", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=7.25, step=0.5)
    sex = st.selectbox("Sex", ["male", "female"])
with col3:
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

if st.button("Predict"):
    try:
        start = time.time()
        row = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }])[FEATURES]

        proba = model.predict_proba(row)[0][1]
        pred = int(proba >= 0.5)
        latency = (time.time() - start) * 1000.0
        latencies_ms.append(latency)

        st.success(f"Predicted Survived: **{pred}** (probability: {proba:.2f})")
        st.write(f"Latency: {latency:.1f} ms")

        logging.info("prediction_ok latency_ms=%.1f proba=%.3f", latency, proba)
    except Exception as e:
        logging.exception("prediction_error")
        st.error(f"Error: {e}")

# Sidebar metrics
st.sidebar.header("Live Metrics")
if len(latencies_ms) > 0:
    p95 = float(np.percentile(latencies_ms, 95))
    st.sidebar.write(f"p95 latency: **{p95:.1f} ms**")
    st.sidebar.write(f"Throughput (since start): **{len(latencies_ms)} requests**")
else:
    st.sidebar.write("No requests yet.")
st.sidebar.caption("For production metrics, also use Azure Monitor & container logs.")
