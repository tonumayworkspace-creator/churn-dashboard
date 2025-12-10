# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/logistic_model.joblib")

# Load training columns to maintain correct order
X_train = pd.read_csv("X_train.csv")
feature_columns = X_train.columns.tolist()

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict whether they are likely to churn.")

# ---- Input fields ----
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# ---- Convert to DataFrame ----
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
})

# Apply one-hot encoding to match training format
input_encoded = pd.get_dummies(input_data)
all_features = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
for col in input_encoded.columns:
    if col in all_features.columns:
        all_features[col] = input_encoded[col].values

# ---- Predict ----
if st.button("Predict Churn"):
    prediction = model.predict(all_features)[0]
    probability = model.predict_proba(all_features)[0][1]

    st.write("Churn Probability:", round(probability, 3))

    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is NOT likely to churn.")
