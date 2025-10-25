from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Credit Risk Prediction API")

# Load trained Random Forest model pipeline
model = joblib.load("credit_risk_model.pkl")

# Define schema
class CreditInput(BaseModel):
    Status_of_existing_checking_account: str
    Duration_in_month: int
    Credit_history: str
    Purpose: str
    Credit_amount: float
    Savings_account_bonds: str
    Present_employment_since: str
    Installment_rate_in_percentage_of_disposable_income: int
    Personal_status_and_sex: str
    Other_debtors_guarantors: str
    Present_residence_since: int
    Property: str
    Age_in_years: int
    Other_installment_plans: str
    Housing: str
    Number_of_existing_credits_at_this_bank: int
    Job: str
    Number_of_people_being_liable_to_provide_maintenance_for: int
    Telephone: str
    Foreign_worker: str

@app.get("/")
def home():
    return {"status": "running", "message": "Credit Risk Model API"}

@app.post("/predict")
def predict_credit(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0, 1]
    result = "Rejected (Bad Risk)" if prediction == 1 else "Approved (Good Risk)"
    return {"Result": result, "Bad Credit Probability": round(float(prob), 3)}
