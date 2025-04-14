import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('loan_approval_model.pkl')
label_encoders = joblib.load('loan_label_encoders.pkl')

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")

with st.form("loan_form"):
    Gender = st.radio("Gender", ["Male", "Female"])
    Married = st.radio("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.radio("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.radio("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term(in days)", min_value=0)
    Credit_History = st.radio("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })

    # Apply label encoding to match the training
    for col in label_encoders:
        if col in input_data.columns:
            le = label_encoders[col]
            input_data[col] = le.transform(input_data[col])

    prediction = model.predict(input_data)[0]
    result = "✅ Approved!" if prediction == 1 else "❌ Rejected."

    st.success(f"Loan Status: {result}")
