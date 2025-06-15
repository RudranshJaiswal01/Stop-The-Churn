import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def load_model():
    return joblib.load("model.pkl")

def display_model_comparison():
    st.markdown("### 🧪 Model Comparison")
    st.markdown("Logistic Regression had the highest AUC-ROC and balanced metrics.")

    data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "MLP"],
        "AUC-ROC": [0.86, 0.84, 0.83, 0.86],
        "F1-Score": [0.78, 0.78, 0.78, 0.80],
        "Precision": [0.77, 0.72, 0.73, 0.76],
        "Recall": [0.79, 0.85, 0.83, 0.84]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # ROC Curve
    st.markdown("#### 📈 ROC Curve")
    st.image("static/roc_curve (1).png")
