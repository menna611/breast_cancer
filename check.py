import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("🧬 Breast Cancer Prediction from Lab Results Image")

# Load data and train model
@st.cache_data
def load_and_train():
   df = pd.read_csv("breast_cancer_data.csv")
  # Make sure this CSV exists
   df = df.dropna()
    
    X = df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1, errors='ignore')
    y = df['diagnosis'].map({'M': 1, 'B': 0})  # 1 = Malignant, 0 = Benign

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns

model, feature_names = load_and_train()

# Upload image
uploaded_file = st.file_uploader("📷 Upload image of lab results", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR to extract text
    extracted_text = pytesseract.image_to_string(image)
    st.subheader("📝 Extracted Text:")
    st.text(extracted_text)

    # Manual parsing section (editable)
    st.subheader("✍️ Confirm or Edit Extracted Values:")

    # Example: Assume the values are space-separated (you may need to adjust)
    try:
        values = [float(val) for val in extracted_text.strip().split() if val.replace('.', '', 1).isdigit()]
        
        if len(values) != len(feature_names):
            st.error(f"❌ Expected {len(feature_names)} features, but got {len(values)} from OCR.")
        else:
            input_data = pd.DataFrame([values], columns=feature_names)
            st.dataframe(input_data)

            # Predict
            prediction = model.predict(input_data)[0]
            result = 'Malignant (M)' if prediction == 1 else 'Benign (B)'
            st.success(f"🧾 Prediction Result: **{result}**")
    except Exception as e:
        st.error(f"⚠️ Could not parse the extracted text: {e}")


