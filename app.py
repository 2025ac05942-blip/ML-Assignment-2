import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Classification Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Classification Model Evaluation")
st.caption("Machine Learning Assignment 2 — interactive test-data evaluation")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib",
}

@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}

models = load_models()

st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV containing the test data and Diagnosis column",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload test_data.csv from the GitHub repository to begin.")
    st.write("Expected target column: `Diagnosis`")
    st.write("Diagnosis encoding: 0 = Benign, 1 = Malignant")
    st.stop()

df = pd.read_csv(uploaded)

if "Diagnosis" not in df.columns:
    st.error("The uploaded CSV must contain a 'Diagnosis' target column.")
    st.stop()

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"].astype(int)

# Ensure only the 30 expected feature columns are used.
expected_features = list(models["Logistic Regression"].feature_names_in_) \
    if hasattr(models["Logistic Regression"], "feature_names_in_") else list(X.columns)

missing = [c for c in expected_features if c not in X.columns]
extra = [c for c in X.columns if c not in expected_features]

if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X = X[expected_features]

st.subheader("2. Test data preview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Rows: **{len(df)}** | Features: **{len(X.columns)}**")

st.sidebar.header("3. Select model")
selected = st.sidebar.selectbox("Classification model", list(models.keys()))

model = models[selected]
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, prob),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}

st.subheader(f"4. Evaluation Metrics — {selected}")
cols = st.columns(6)
for col, (metric, value) in zip(cols, metrics.items()):
    col.metric(metric, f"{value:.4f}")

st.subheader("5. Confusion Matrix")
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(figsize=(4.5, 3.5))
im = ax.imshow(cm)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks([0, 1], ["Benign", "Malignant"])
ax.set_yticks([0, 1], ["Benign", "Malignant"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im, ax=ax)
st.pyplot(fig, clear_figure=True)

st.subheader("6. Classification Report")
report = classification_report(
    y, pred, target_names=["Benign", "Malignant"], output_dict=True
)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("7. Prediction Output")
output = df.copy()
output["Predicted_Diagnosis"] = pred
output["Malignant_Probability"] = prob.round(4)
st.dataframe(output, use_container_width=True)

csv = output.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions CSV",
    csv,
    "predictions.csv",
    "text/csv"
)

st.markdown("---")
st.caption("Dataset: UCI Breast Cancer Wisconsin (Diagnostic). "
           "This app evaluates the uploaded test data using saved trained models.")
