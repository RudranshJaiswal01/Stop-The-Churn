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
        "AUC-ROC": [0.84, 0.81, 0.83, 0.79],
        "F1-Score": [0.75, 0.71, 0.74, 0.68],
        "Precision": [0.76, 0.70, 0.74, 0.66],
        "Recall": [0.74, 0.72, 0.73, 0.70]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # ROC Curve
    st.markdown("#### 📈 ROC Curve")
    st.markdown("""
        <div style='text-align: center;'>
            <img src='static/roc_curve.png' width='600'/>
        </div>
        """, unsafe_allow_html=True)

