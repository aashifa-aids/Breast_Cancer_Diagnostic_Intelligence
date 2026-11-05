from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Diagnostic ML Demonstration", page_icon="🔬", layout="wide")
st.title("Breast Cancer Diagnostic ML Demonstration")
st.warning(
    "Educational demonstration only. This is not a medical device, does not diagnose cancer, "
    "and must not guide healthcare decisions."
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model files are missing. Run `python run_project.py` from the project root first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())
features = metadata["feature_names"]
defaults = {
    "mean radius": 14.0, "mean texture": 20.0, "mean perimeter": 90.0,
    "mean area": 600.0, "mean smoothness": 0.10, "mean compactness": 0.10,
    "mean concavity": 0.08, "mean concave points": 0.05,
    "mean symmetry": 0.18, "mean fractal dimension": 0.06
}

st.write("Enter numeric values to see an educational model output.")
with st.form("prediction_form"):
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(features):
        with columns[i % 3]:
            values[feature] = st.number_input(
                feature, value=float(defaults.get(feature, 0.0)), format="%.6f"
            )
    submitted = st.form_submit_button("Run demonstration prediction")

if submitted:
    row = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    predicted = int(model.predict(row)[0])
    label = "Malignant class (M)" if predicted == 1 else "Benign class (B)"
    st.subheader("Model output")
    st.write(f"Predicted dataset class: **{label}**")
    st.write(f"Model score for malignant class: **{probability:.3f}**")
    st.caption(
        "This score is not a personal cancer risk or calibrated clinical probability. "
        "It is not a diagnosis. For medical concerns, consult a qualified healthcare professional."
    )
