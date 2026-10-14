"""Simple Streamlit app for student score prediction."""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "linear_regression.pkl"
RESULTS_PATH = ROOT / "reports" / "model_results.json"
CHART_PATH = ROOT / "reports" / "model_comparison.png"

st.set_page_config(page_title="Student Score Predictor", page_icon="📚")
st.title("📚 Student Performance Prediction")
st.write(
    "Enter a few study-related details to get a sample predicted final score. "
    "This is a beginner learning project, not a real assessment tool."
)

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

study_hours = st.slider("Study hours per day", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=75, step=1)
previous_score = st.slider("Previous score", min_value=0, max_value=100, value=60, step=1)

if st.button("Predict score"):
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
    }])
    prediction = float(model.predict(row)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.success(f"Estimated final score: {prediction:.1f} / 100")
    st.caption("This estimate comes from a small illustrative sample dataset.")

st.divider()
st.subheader("Model results")
if RESULTS_PATH.exists():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame(results).T, use_container_width=True)
if CHART_PATH.exists():
    st.image(str(CHART_PATH), caption="Mean Absolute Error comparison", use_container_width=True)

st.subheader("About this project")
st.write(
    "The project compares Linear Regression and Decision Tree Regression. "
    "MAE measures average prediction error; lower is better. R² summarizes how "
    "well predictions match the test values, but it can be unstable on tiny datasets."
)
