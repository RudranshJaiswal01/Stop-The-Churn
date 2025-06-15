import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_data, predict_churn, plot_churn_distribution, plot_probability_bars
from model_comparison import display_model_comparison, load_model
import shap
import joblib
from io import BytesIO

st.set_page_config(page_title="Churn Predictor Dashboard", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")

# ---------------------- Section: Model Summary ----------------------
st.markdown("""
### 🔍 Model Overview and Performance
This dashboard uses **Logistic Regression** to predict churn. Multiple models were evaluated and compared. Logistic Regression was selected due to its high AUC-ROC score, interpretability, and balanced performance.
""")
display_model_comparison()

# ---------------------- Section: Upload + Predict ----------------------
st.markdown("---")
st.markdown("""
### 📁 Upload Customer Data for Prediction
Upload a `.csv` file containing customer records. Predictions will be generated using the pre-trained model.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Sample of Uploaded Data")
    st.write(df.head())

    model = load_model()
    df_processed, indices = preprocess_data(df.copy())
    df_with_preds = predict_churn(df, df_processed, indices, model)

    threshold = st.slider("🔧 Set Churn Threshold", min_value=0.0, max_value=1.0, value=0.4921, step=0.01)
    df_with_preds['predicted_churn'] = (df_with_preds['churn_probability'] > threshold).astype(int)

    st.success("✅ Predictions Generated")

    st.download_button("📥 Download Predictions CSV", data=BytesIO(df_with_preds.to_csv(index=False).encode()),
                       file_name='predictions.csv', mime='text/csv')

    st.markdown("---")
    st.subheader("📊 Prediction Insights")
    plot_churn_distribution(df_with_preds)
    plot_probability_bars(df_with_preds)

    st.subheader("🚨 Top 10 High-Risk Customers")
    top_10 = df_with_preds.sort_values(by='churn_probability', ascending=False).head(10)
    st.dataframe(top_10)

    # SHAP Analysis
    st.markdown("---")
    st.subheader("🧠 SHAP Explanation for Model Predictions")
    explainer = shap.Explainer(model, df_processed)
    shap_values = explainer(df_processed)
    st_shap = st.container()
    with st_stap:
        st.markdown("#### Feature Importance Summary Plot")
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
