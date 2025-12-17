# Churn Prediction Model

📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn, along with churn probability.
The project includes data preprocessing, model training, evaluation, and deployment using Streamlit.

🚀 Project Overview

Customer churn has a direct impact on revenue in subscription-based businesses.
This project aims to predict customer churn in advance, enabling businesses to take proactive retention actions.

The solution:

Trains machine learning models on historical customer data

Predicts churn probability for new customers

Provides a user-friendly web interface for real-time predictions

🎯 Business Problem

Retaining customers is cheaper than acquiring new ones

High churn leads to revenue loss

Predicting churn helps businesses:

Target high-risk customers

Offer personalized discounts

Improve service quality

📁 Dataset

Telco Customer Churn Dataset

~7,000 customer records

Features include:

Demographics (gender, senior citizen)

Services (internet, streaming, security)

Contract type

Billing and payment methods

Target variable: Churn (Yes / No)

🧹 Data Preprocessing

Key preprocessing steps:

Converted TotalCharges from object to numeric

Handled missing values

Dropped irrelevant column (customerID)

Identified numerical and categorical features

Encoded categorical features using Label Encoding

Saved label mappings for consistent deployment

Ensured feature alignment using model.feature_names_in_

🧠 Model Building

Models trained and evaluated:

Logistic Regression (baseline)

Random Forest (final model)

XGBoost (experimented)

Why Random Forest?

Performs well on tabular data

Captures non-linear relationships

Robust to feature scaling

Provides feature importance

📈 Model Evaluation

Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Since churn is an imbalanced classification problem, emphasis was placed on recall for churn customers.

🔢 Probability-Based Prediction

Instead of only predicting churn/no-churn:

Used predict_proba() to output churn probability

Enables business-driven thresholds (e.g., 40% instead of 50%)

Example output:

Low Churn Risk (38%)

🖥️ Deployment (Streamlit App)

The trained model is deployed using Streamlit.

App features:

Text-based dropdowns (user-friendly)

Real-time churn prediction

Churn probability display

Risk classification (Low / High)

Files used:

churn_model.pkl → trained model

label_mappings.pkl → label encoding mappings

app.py → Streamlit UI

🧩 Challenges & Solutions
Challenge	Solution
Feature mismatch during prediction	Used model.feature_names_in_
Label encoding inconsistency	Saved and reused label mappings
UI showing numeric values	Implemented text → numeric mapping
Deployment errors	Ensured consistent preprocessing
🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Joblib

Streamlit

🏗️ Project Architecture
User (Streamlit UI)
        |
        v
Input Processing & Encoding
        |
        v
Trained ML Model (Random Forest / XGBoost)
        |
        v
Churn Prediction & Probability
        |
        v
Result Display (Low / High Risk)

▶️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Install dependencies

pip install -r requirements.txt


Run Streamlit app

python -m streamlit run app.py

✅ Final Outcome

The system predicts:

Whether a customer will churn

Probability of churn

This allows businesses to take data-driven retention decisions.

📌 Key Learnings

Importance of feature consistency in ML deployment

Handling categorical encoding across training and inference

Building end-to-end ML applications

Bridging ML models with user-facing interfaces

📬 Contact

If you have any questions or suggestions, feel free to reach out.

⭐ If you like this project, give it a star!
