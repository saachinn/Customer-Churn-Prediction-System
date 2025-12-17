import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model and encoders
# =========================
model = joblib.load("churn_model.pkl")
label_maps = joblib.load("label_mappings.pkl")

# =========================
# App title
# =========================
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability")

# =========================
# Numeric inputs
# =========================
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# =========================
# Categorical inputs (TEXT → NUMBER)
# =========================
contract_text = st.selectbox(
    "Contract Type",
    list(label_maps["Contract"].keys())
)
contract = label_maps["Contract"][contract_text]

internet_text = st.selectbox(
    "Internet Service",
    list(label_maps["InternetService"].keys())
)
internet = label_maps["InternetService"][internet_text]

payment_text = st.selectbox(
    "Payment Method",
    list(label_maps["PaymentMethod"].keys())
)
payment = label_maps["PaymentMethod"][payment_text]

# =========================
# Create input dataframe
# MUST match training columns
# =========================
input_dict = {
    "gender": 0,      # default
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": tenure,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": internet,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": contract,
    "PaperlessBilling": 1,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges


}

# VERY IMPORTANT: enforce exact feature order
input_data = pd.DataFrame([input_dict])[model.feature_names_in_]

# =========================
# Prediction
# =========================
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Churn Risk ⚠️ ({probability:.2%})")
    else:
        st.success(f"Low Churn Risk ✅ ({probability:.2%})")
