import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cyber ML - Sriviveka", page_icon="🛡️", layout="wide")

st.title("🛡️ Cyber Intrusion Detection - ML Assignment 2")
st.write("Hello Sriviveka 😄 — model comparison + predictions.")

# -----------------------------
# A) Model comparison table
# -----------------------------
st.subheader("📊 Model Comparison Table (sorted by AUC)")

try:
    df_compare = pd.read_csv("model/model_comparison.csv")
    st.dataframe(df_compare, use_container_width=True)
except Exception as e:
    st.warning("Could not load model/model_comparison.csv")
    st.code(str(e))

st.divider()

# -----------------------------
# B) Load trained model pipeline
# -----------------------------
st.subheader("🔍 Predict Attack vs Normal from CSV")

try:
    model = joblib.load("model/best_random_forest.joblib")
except Exception as e:
    st.error("❌ Could not load model/best_random_forest.joblib")
    st.code(str(e))
    st.stop()

# Threshold slider
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

    # Keep a copy for evaluation if ground truth exists
    y_true = None
    if "label" in data.columns:
        y_true = data["label"].copy()

    # Drop label/attack_cat if present (so model doesn't use answer key)
    drop_if_present = [c for c in ["label", "attack_cat"] if c in data.columns]
    X = data.drop(columns=drop_if_present) if drop_if_present else data.copy()

    # -----------------------------
    # Column mismatch safety
    # -----------------------------
    # Get feature names expected by the preprocessor (if available)
    expected_cols = None
    try:
        preprocessor = model.named_steps.get("prep", None)
        if hasattr(preprocessor, "feature_names_in_"):
            expected_cols = list(preprocessor.feature_names_in_)
    except Exception:
        expected_cols = None

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in X.columns]
        extra = [c for c in X.columns if c not in expected_cols]

        if missing:
            st.error("❌ Uploaded CSV is missing required columns.")
            st.write("Missing columns:")
            st.code(missing)
            st.write("Tip: Upload the original UNSW_NB15_testing-set.csv (or a subset of it).")
            st.stop()

        # Reorder columns to match training
        X = X[expected_cols]

        if extra:
            st.info("ℹ️ Your CSV had extra columns. They were ignored.")
            st.code(extra)

    # -----------------------------
    # Predict
    # -----------------------------
    try:
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        st.error("❌ Prediction failed. This usually happens due to column mismatch or data types.")
        st.code(str(e))
        st.stop()

    # Results
    result = X.copy()
    result["Risk_Score"] = probs
    result["Prediction"] = ["Attack" if p == 1 else "Normal" for p in preds]

    st.success("✅ Done! Showing first 50 predictions 😄")
    st.dataframe(result.head(50), use_container_width=True)

    # Summary counts
    st.subheader("🧾 Summary")
    attack_count = int((preds == 1).sum())
    normal_count = int((preds == 0).sum())
    st.write(f"**Attack:** {attack_count}  |  **Normal:** {normal_count}")

    # Download button
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=result.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------
    # C) Evaluation (ONLY if true label exists)
    # -----------------------------
    st.subheader("📐 Evaluation (only if uploaded CSV contains true labels)")

    if y_true is None:
        st.info("Your uploaded CSV does not contain a `label` column, so confusion matrix can't be computed. 😄")
        st.write("If you upload the original UNSW_NB15_testing-set.csv (with `label`), you'll see evaluation here.")
    else:
        # Confusion matrix: y_true vs preds
        cm = confusion_matrix(y_true, preds)

        st.write("✅ Confusion Matrix (True label vs Predicted)")
        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # show values inside boxes
        for (i, j), val in pd.DataFrame(cm).stack().items():
            ax.text(j, i, str(val), ha="center", va="center")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal(0)", "Attack(1)"])
        ax.set_yticklabels(["Normal(0)", "Attack(1)"])

        st.pyplot(fig)

        st.write("✅ Classification Report")
        st.code(classification_report(y_true, preds))
