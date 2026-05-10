import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# Custom CSS for colors
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("naive_bayes_Dibetesmodel.pkl")

# Title
st.title("🩺 Diabetes Prediction Dashboard")
st.write("Enter patient details to predict diabetes")

# Layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    blood_pressure = st.number_input("Blood Pressure", 0, 150)

with col2:
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

# Predict button
if st.button("🔍 Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi,
                          dpf, age]])

    prediction = model.predict(features)

    # Output
    if prediction[0] == 1:
        st.markdown(
            '<div class="result-box" style="background-color:#ff4d4d;color:white;">⚠️ High Risk of Diabetes</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#4CAF50;color:white;">✅ Low Risk of Diabetes</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.write("💡 Built with Streamlit | ML Model: Naive Bayes")