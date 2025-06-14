import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import BytesIO

# Streamlit Config
st.set_page_config(page_title="Churn Predictor Dashboard", layout="wide")

# Dark mode toggle
dark_mode = st.toggle("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
            body { background-color: #0e1117; color: #ffffff; }
            .stApp { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)

# Title
st.title("📉 Churn Prediction Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File with Customer Data", type=["csv"])

def get_sparse_encoded_unseen(df):
    df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['StreamingTV'] = df['StreamingTV'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['TechSupport'] = df['TechSupport'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 0, 'No': 1, 'No internet service': 2})
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'Two year': 1, 'One year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Credit card (automatic)': 2, 'Bank transfer (automatic)': 3})
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    return df

def predict(df):
    model = joblib.load("model.pkl")
    X = df.copy()
    X = X.drop(columns=['customerID', 'TotalCharges'], errors='ignore')
    X['MonthlyCharges'] = X['MonthlyCharges'].astype(float)
    null_indices = X[X.isnull().any(axis=1)].index
    X_okay_indices = X.index.difference(null_indices)
    X_not_okay = X.loc[null_indices]
    X_okay = X.loc[X_okay_indices]
    X_okay = get_sparse_encoded_unseen(X_okay)
    probs = model.predict_proba(X_okay)[:, 1]
    df['churn_probability'] = np.nan
    df.loc[X_okay_indices, 'churn_probability'] = probs
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Sample")
    st.write(df.head())

    df = predict(df.copy())
    df['predicted_churn'] = (df['churn_probability'] > 0.4921).astype(int)

    st.success("✅ Predictions Generated Using Pre-trained Model")

    # Download predictions
    def convert_df(df):
        output = BytesIO()
        df.to_csv(output, index=False)
        return output.getvalue()

    st.download_button("📥 Download Predictions CSV", data=convert_df(df), file_name='predictions.csv', mime='text/csv')

    # Histogram of Churn Probability
    st.subheader("🔍 Churn Probability Distribution")
    fig1, ax1 = plt.subplots(figsize=(4, 2))
    sns.histplot(df['churn_probability'], bins=20, kde=True, color='skyblue', ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1)

    # Pie Chart
    st.subheader("📊 Churn vs Retain Distribution")
    pie_data = df['predicted_churn'].value_counts().rename({0: 'Retain', 1: 'Churn'})
    fig3, ax3 = plt.subplots(figsize=(2,2))
    ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax3.axis('equal')
    plt.tight_layout()
    st.pyplot(fig3)

    # Top 10 High Risk Customers
    st.subheader("🚨 Top 10 High-Risk Customers")
    high_risk = df.sort_values(by='churn_probability', ascending=False).head(10)
    st.dataframe(high_risk)

    # Bar Chart by Risk Level
    st.subheader("🎯 Churn Probability Bar Chart")
    display_df = df[['churn_probability']].copy()
    bar_colors = display_df['churn_probability'].apply(
        lambda x: 'red' if x > 0.7 else ('yellow' if x > 0.35 else 'green')
    )
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.bar(display_df.index, display_df['churn_probability'], color=bar_colors)
    ax2.set_ylabel('Probability to Churn')
    ax2.set_xlabel('Customer Index (Sorted)')
    plt.tight_layout()
    st.pyplot(fig2)
