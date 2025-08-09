import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Breast Cancer ML App", layout="wide")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_default_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    df = pd.concat([X, y.rename("target")], axis=1)
    return df, data

def preprocess(df, target_col="target", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler

@st.cache_data
def train_models(X_train, y_train):
    models = {}
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Save models
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    return models

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": roc,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------- UI --------------------
st.title("Breast Cancer — Decision Tree & Random Forest")
st.write("Train, compare, and use models for breast cancer classification.")

with st.sidebar:
    st.header("Data & Model Options")
    data_source = st.radio("Dataset source", ["Built-in (sklearn)", "Upload CSV"])
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    test_size = st.slider("Test set size (%)", 10, 50, 20)
    test_size_frac = test_size / 100
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    if st.button("Clear saved models"):
        for f in os.listdir(MODEL_DIR):
            try:
                os.remove(os.path.join(MODEL_DIR, f))
            except:
                pass
        st.success("Models cleared")

# -------------------- LOAD DATA --------------------
if data_source == "Upload CSV" and uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df, _ = load_default_dataset()

if "target" not in df.columns:
    st.error("Dataset must contain a 'target' column (0 = benign, 1 = malignant).")
    st.stop()

st.write(f"Dataset shape: {df.shape}")
if st.checkbox("Show data sample"):
    st.dataframe(df.head())

# -------------------- TRAIN MODELS --------------------
with st.spinner("Training models..."):
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = preprocess(
        df, test_size=test_size_frac, random_state=random_state
    )
    models = train_models(X_train, y_train)

# -------------------- EVALUATION --------------------
st.header("Model Comparison")
eval_list = []
for name, model in models.items():
    ev = evaluate_model(model, X_test, y_test, name)
    eval_list.append(ev)

eval_df = pd.DataFrame(eval_list).sort_values("accuracy", ascending=False)
st.dataframe(eval_df[['name','accuracy','precision','recall','f1','roc_auc']].round(3))

# Confusion matrix for best model
best_model_name = eval_df.iloc[0]['name']
best_model = models[best_model_name]
st.subheader(f"Best Model: {best_model_name}")
st.pyplot(plot_confusion_matrix(eval_df.iloc[0]['confusion_matrix']))

# -------------------- FEATURE IMPORTANCE --------------------
st.header("Feature Importance")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    st.bar_chart(fi.head(15))

# -------------------- PREDICTION --------------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Single Prediction")
    input_vals = {}
    cols = X_train.columns
    c1, c2 = st.columns(2)
    for i, col in enumerate(cols):
        if i % 2 == 0:
            with c1:
                input_vals[col] = st.number_input(col, value=float(X_train[col].median()))
        else:
            with c2:
                input_vals[col] = st.number_input(col, value=float(X_train[col].median()))

    input_df = pd.DataFrame([input_vals])
    chosen_model = st.selectbox("Choose model", list(models.keys()))

    if st.button("Predict"):
        model = models[chosen_model]
        proba = model.predict_proba(input_df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        st.metric("Prediction", "Malignant" if pred[0] else "Benign")
        st.write(f"Probability of malignant: {proba[0]:.3f}")

with tab2:
    st.subheader("Batch Prediction")
    csv_file = st.file_uploader("Upload CSV (no target column)", type=["csv"])
    model_choice = st.selectbox("Model", list(models.keys()))
    if csv_file is not None:
        df_batch = pd.read_csv(csv_file)
        if set(X_train.columns) - set(df_batch.columns):
            st.error("CSV is missing required features.")
        else:
            df_batch = df_batch[X_train.columns]
            if st.button("Run Batch Prediction"):
                model = models[model_choice]
                probs = model.predict_proba(df_batch)[:, 1]
                preds = (probs >= 0.5).astype(int)
                out = df_batch.copy()
                out["prob_malignant"] = probs
                out["prediction"] = preds
                st.dataframe(out.head(50))
                st.download_button("Download Predictions", data=df_to_csv_bytes(out), file_name="predictions.csv")

