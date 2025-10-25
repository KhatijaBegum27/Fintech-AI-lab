import os, json, warnings   
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import zipfile
import os

zip_path = "statlog+german+credit+data.zip"
extract_path = "C:\\Users\\Khatija Begum\\Downloads\\WDI_Data\\CreditRisk"
df = pd.read_csv("CreditRisk\german.data", 
                 sep=r'\s+', header=None)

# DATA EXPLORATION
print("First 5 rows:")
print(df.head())

print("Shape:", df.shape)

print("\n Info:")
print(df.info())

print("\n Summary Statistics:")
print(df.describe(include='all'))

# DATA CLEANING

columns = [
    "Status_of_existing_checking_account", "Duration_in_month",
    "Credit_history", "Purpose", "Credit_amount",
    "Savings_account_bonds", "Present_employment_since",
    "Installment_rate_in_percentage_of_disposable_income",
    "Personal_status_and_sex", "Other_debtors_guarantors",
    "Present_residence_since", "Property", "Age_in_years",
    "Other_installment_plans", "Housing",
    "Number_of_existing_credits_at_this_bank", "Job",
    "Number_of_people_being_liable_to_provide_maintenance_for",
    "Telephone", "Foreign_worker", "Target"
]
df.columns = columns

print("\n Columns:")
print(df.columns.tolist())

# Map target variable
# There are 20 features + 1 target
# Columns 0–19 = features, 20 = target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# The dataset uses 1 = good, 2 = bad
y = y.map({1: 0, 2: 1})   # 0 = good, 1 = bad

print("\nTarget distribution:\n", df["Target"].value_counts())


# PREPROCESSING PIPELINE

num_cols = [i for i in X.columns if X[i].dtype != 'object']
cat_cols = [i for i in X.columns if X[i].dtype == 'object']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# MODEL PIPELINE

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=5000,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
])


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN MODEL

print("\n Training model...")
model.fit(X_train, y_train)
print(" Model trained successfully!")


#  EVALUATION

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n MODEL PERFORMANCE:")
print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1 Score  : {f1:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))



#  SAVE TRAINED MODEL

joblib.dump(model, "credit_risk_model.pkl")
print("\n Model saved as credit_risk_model.pkl")