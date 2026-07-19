# MSc in Computer Science
# Batch        : Feb 2026 Batch
# Student Name : ZUBAIR MARIKKAR MOHAMED NUZARATH
# Module       : COM 763 Advanced Machine Learning
# Assignment   : Part 01 - Portfolio Task
# Description  : Design, build, evaluate, and demonstrate a
#                complete machine learning system that solves
#                a real-world problem
#                Streamlit Community Cloud Deployment        
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading model and scaler - these were generated after training the model
# -on given dataset
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

st.set_page_config(page_title="ML Powered - Loan Approval Prediction", page_icon="💡")

# Loading and display banner image
top_banner = Image.open('banner.jpg')
st.image(top_banner, use_container_width=True)

st.title("Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval.")
st.write("Make sure you fill the fields properly")

# Creating the Input fields to capture User Input
gender = st.selectbox("Gender", ["Female", "Male"])
married = st.selectbox("Marital Status", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

applicantincome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
coapplicantincome = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
loanamount = st.number_input("Loan Amount", min_value=0.0, value=150.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)

credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Encoding inputs done below
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0

dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
dependents = dep_map[dependents]

education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
property_area = area_map[property_area]

# Prediction button to trigger the Prediction process
if st.button("Predict Loan Status"):

    sample = pd.DataFrame({
        "gender": [gender],
        "married": [married],
        "dependents": [dependents],
        "education": [education],
        "self_employed": [self_employed],
        "applicantincome": [applicantincome],
        "coapplicantincome": [coapplicantincome],
        "loanamount": [loanamount],
        "loan_amount_term": [loan_amount_term],
        "credit_history": [credit_history],
        "property_area": [property_area]
    })

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    if prediction == "y":
        st.success("Loan Approved - 😊 Congratulations")
    else:
        st.error("Loan Rejected - 😢 Sorry")

    
    st.write("Web Solution Developed by Mohamed Nuzarath - s25026165 | Wrexham University | Bacth 2026")