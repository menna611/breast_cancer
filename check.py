import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Setup and Training
# -------------------------------
df = pd.read_csv("C:\\Users\\Incorta\\Desktop\\breast_cancer\\breast_cancer_data (1).csv")

# Encode 'diagnosis': M -> 1, B -> 0
if df['diagnosis'].dtype == object:
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Drop irrelevant columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
# Page config
st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="wide")

# Sidebar
st.sidebar.title("ℹ️ About the App")
st.sidebar.markdown("""
This app uses a **Decision Tree Classifier** to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on lab results.  
- Built with ❤️ using Streamlit  
- Powered by scikit-learn  
""")

st.sidebar.write("📊 Features used:")
st.sidebar.write(list(X.columns))

# Title
st.markdown("<h1 style='text-align: center; color: #5C4B99;'>🔬 Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter lab values below to get an instant diagnosis prediction.</p>", unsafe_allow_html=True)

st.divider()

# Input form
st.subheader("📥 Enter Lab Results")
user_input = {}
cols = st.columns(3)  # arrange inputs in 3 columns

for i, col in enumerate(X.columns):
    with cols[i % 3]:
        user_input[col] = st.number_input(f"{col}", value=float(round(X[col].mean(), 2)))

# Prediction
if st.button("🧪 Predict Diagnosis"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "Malignant (M)" if prediction == 1 else "Benign (B)"

    # Output
    st.markdown("### 🧾 Prediction Result")
    if prediction == 1:
        st.error("🚨 The tumor is **Malignant (M)**. Please consult a specialist.")
    else:
        st.success("✅ The tumor is **Benign (B)**. No signs of malignancy detected.")

    with st.expander("🔎 View Input Summary"):
        st.dataframe(input_df.style.format("{:.2f}"), use_container_width=True)

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>© 2025 - Breast Cancer Predictor | Built with Streamlit</p>",
    unsafe_allow_html=True
)

