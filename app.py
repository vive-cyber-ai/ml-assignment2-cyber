import streamlit as st
import pandas as pd
import joblib

# ------------------------------------
# Page settings
# ------------------------------------
st.set_page_config(page_title="Cyber ML - Vive", page_icon="🛡️", layout="wide")

st.title("🛡️ Cyber Intrusion Detection - ML Assignment 2")
st.write("Hello Sriviveka 😄 — this app shows model comparison + predictions.")

# ------------------------------------
# A) Model comparison table
# ------------------------------------
st.subheader("📊 Model Comparison Table (sorted by AUC)")

df = pd.read_csv("model/model_comparison.csv")
st.dataframe(df, use_container_width=True)

st.divider()

# ------------------------------------
# B) Prediction section
# ------------------------------------
st.subheader("🔍 Predict Attack vs Normal from CSV")

# Load saved best model pipeline (preprocess + classifier)
model = joblib.load("model/best_random_forest.joblib")

# Threshold slider (lets you control strictness)
threshold = st.slider(
    "Risk threshold (higher = fewer alarms, lower = catch more attacks) 😄",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

uploaded = st.file_uploader("Upload a CSV file (same columns as dataset)", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)

    # If user uploads full dataset (with labels), remove them safely
    for c in ["label", "attack_cat"]:
        if c in data.columns:
            data = data.drop(columns=[c])

    # Predict probabilities
    probs = model.predict_proba(data)[:, 1]

    # Apply threshold to convert probability -> Attack/Normal
    preds = (probs >= threshold).astype(int)

    # Build result table
    result = data.copy()
    result["Risk_Score"] = probs
    result["Prediction"] = ["Attack" if p == 1 else "Normal" for p in preds]

    st.success("✅ Done! Showing first 50 predictions 😄")
    st.dataframe(result.head(50), use_container_width=True)

    # Summary counts
    st.subheader("🧾 Summary")
    attack_count = int((preds == 1).sum())
    normal_count = int((preds == 0).sum())
    st.write(f"**Attack:** {attack_count}  |  **Normal:** {normal_count}")

    # Download predictions
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=result.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
