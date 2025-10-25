import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title(" Loan Credit Risk Prediction ")

API_URL = "http://127.0.0.1:8000/predict"

st.subheader("Enter Applicant Details")

age = st.number_input("Age (years)", 18, 75, 30)
duration = st.slider("Duration (months)", 4, 72, 24)
credit_amount = st.number_input("Credit Amount", 500, 20000, 3000)
housing = st.selectbox("Housing", ["own", "rent", "free"])
job = st.selectbox("Job", ["skilled", "unskilled", "management", "student"])
purpose = st.selectbox("Purpose", ["car", "education", "furniture", "business"])
status = st.selectbox("Checking Account", ["A11", "A12", "A13", "A14"])
credit_history = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
foreign_worker = st.selectbox("Foreign Worker", ["A201", "A202"])

if st.button("Predict Credit Risk"):
    payload = {
        "Status_of_existing_checking_account": status,
        "Duration_in_month": duration,
        "Credit_history": credit_history,
        "Purpose": purpose,
        "Credit_amount": credit_amount,
        "Savings_account_bonds": "A65",
        "Present_employment_since": "A73",
        "Installment_rate_in_percentage_of_disposable_income": 2,
        "Personal_status_and_sex": "A93",
        "Other_debtors_guarantors": "A101",
        "Present_residence_since": 2,
        "Property": "A121",
        "Age_in_years": age,
        "Other_installment_plans": "A143",
        "Housing": housing,
        "Number_of_existing_credits_at_this_bank": 1,
        "Job": job,
        "Number_of_people_being_liable_to_provide_maintenance_for": 1,
        "Telephone": "A191",
        "Foreign_worker": foreign_worker
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(result["Result"])
            st.metric("Bad Credit Probability", f"{result['Bad Credit Probability']*100:.2f}%")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection failed: {e}")