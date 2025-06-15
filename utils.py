import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def preprocess_data(df):
    indices = df.index
    df = df.drop(columns=['customerID', 'TotalCharges'], errors='ignore')
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df = df.dropna()

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
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1,
                                                   'Credit card (automatic)': 2, 'Bank transfer (automatic)': 3})
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    return df, indices

def predict_churn(original_df, processed_df, indices, model):
    probs = model.predict_proba(processed_df)[:, 1]
    original_df['churn_probability'] = None
    original_df.loc[indices, 'churn_probability'] = probs
    return original_df

def plot_churn_distribution(df):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['churn_probability'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Churn Probability")
    st.pyplot(fig)

def plot_probability_bars(df):
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

def plot_pie_chart(df):
    st.subheader("📊 Churn vs Retain Distribution")
    pie_data = df['predicted_churn'].value_counts().rename({0: 'Retain', 1: 'Churn'})
    fig3, ax3 = plt.subplots(figsize=(10,10))
    fig3, ax3 = plt.subplots(figsize=(5,5))
    ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax3.axis('equal')
    plt.tight_layout()
    st.pyplot(fig3)
