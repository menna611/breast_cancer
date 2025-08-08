import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 📝 Title
st.title("🧬 Breast Cancer Prediction from Lab Results Image")
st.write("Upload an image of lab results and the app will predict whether the case is **Malignant (M)** or **Benign (B)**.")

# 📦 Load dataset and train model
@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, data.feature_names

model, feature_names = train_model()

# 📤 Upload image
uploaded_file = st.file_uploader("📷 Upload image of lab results", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔍 OCR to extract text
    extracted_text = pytesseract.image_to_string(image)
    st.subheader("📝 Extracted Text:")
    st.text(extracted_text)

    # ✍️ Let user adjust values
    st.subheader("✏️ Enter or correct extracted values")
    values_input = st.text_area(
        "Paste the numbers here (separated by spaces):",
        value=" ".join([str(round(x, 2)) for x in np.random.rand(len(feature_names))])
    )

    try:
        values = [float(val) for val in values_input.strip().split()]
        if len(values) != len(feature_names):
            st.error(f"❌ Expected {len(feature_names)} values, but got {len(values)}.")
        else:
            input_df = pd.DataFrame([values], columns=feature_names)
            prediction = model.predict(input_df)[0]
            result = "Benign (B)" if prediction == 1 else "Malignant (M)"
            st.success(f"🧾 Prediction Result: **{result}**")
    except ValueError:
        st.error("⚠️ Please enter only numbers separated by spaces.")
