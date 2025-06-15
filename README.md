# 📊 Customer Churn Prediction Dashboard

A Streamlit-based interactive dashboard to predict customer churn using a Logistic Regression model. This project includes an in-depth analysis notebook, a trained model, custom visualization components, and an interactive web interface.

---

## 🔍 Project Highlights

* **Model Used**: Logistic Regression (selected based on best AUC-ROC score)
* **Notebook**: Complete model development, EDA, training, evaluation, and SHAP explanations
* **Interactive Dashboard**: Upload data, set churn thresholds, and visualize insights
* **Model Interpretability**: SHAP plots to explain predictions
* **Custom Visuals**: Histogram, bar chart, and pie chart for churn analysis
* **Live Dashboard**: [Streamlit App Deployment](https://stop-the-churn-by-rudransh.streamlit.app/)

---

## 📁 Folder Structure

```
.
├── app.py                        # Main Streamlit application
├── utils.py                      # Utility functions for preprocessing and visualization
├── model_comparison.py           # Model loading and evaluation comparison display
├── config.toml                   # Streamlit dashboard configuration
├── requirements.txt              # Project dependencies
├── model.pkl                     # Pre-trained Logistic Regression model (required for predictions)
├── Stop_The_Churn_models.ipynb   # Complete Jupyter notebook with EDA and model training
├── static/
│   └── roc_curve (1).png         # ROC curve used in dashboard
```

---

## 📚 Jupyter Notebook Overview

**Notebook**: `Stop_The_Churn_models.ipynb`
**Google Colab**: [Open in Colab](https://colab.research.google.com/drive/1mXZF2xIGM7iZeQA4Ue-uCqTGekTOr1ej?usp=sharing)

This notebook includes:

* Exploratory Data Analysis (EDA) on Telco dataset
* Feature engineering and data preprocessing
* Training and comparison of multiple models:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * MLP (Neural Net)
* Hyperparameter tuning
* ROC Curve, AUC Score, F1-Score comparisons
* Final model selection: **Logistic Regression**
* SHAP values for interpretability

This notebook provides full context behind model selection and rationale.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stop-the-churn.git
cd stop-the-churn
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Pre-trained Model

Make sure `model.pkl` is present in the root directory.

### 5. Launch the App

```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Summary

Models were compared using metrics such as AUC-ROC, F1, Precision, and Recall:

| Model               | AUC-ROC | F1-Score | Precision | Recall |
| ------------------- | ------- | -------- | --------- | ------ |
| Logistic Regression | 0.86    | 0.78     | 0.77      | 0.79   |
| Random Forest       | 0.84    | 0.78     | 0.72      | 0.85   |
| XGBoost             | 0.83    | 0.78     | 0.73      | 0.83   |
| MLP                 | 0.86    | 0.80     | 0.76      | 0.84   |

*Visual: ROC Curve is embedded in the Streamlit app.*

---

## 📄 How to Use

1. Launch the app using `streamlit run app.py`
2. Upload a `.csv` file containing customer data
3. Adjust the churn probability threshold slider
4. Review churn predictions, download results, and explore visualizations
5. Examine top 10 high-risk customers and SHAP summary plots

---

## 🚀 Features

* Threshold slider for prediction classification
* SHAP-based explanation of predictions
* Downloadable prediction results
* Custom Seaborn/Matplotlib visualizations:

  * Histogram of churn probabilities
  * Color-coded bar chart
  * Retain vs. Churn pie chart

---

## ⚖️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **ML Libraries**: scikit-learn, XGBoost, SHAP
* **Visualization**: Matplotlib, Seaborn

---

## 🚧 To Do / Improvements

* Add unit tests for preprocessing and model inference
* Add support for multiple model uploads

---

## 💼 License

MIT License. See `LICENSE`.

---

## ✍️ Author

**Rudransh Jaiswal** &#x20;
Made with ❤️ for applied data science.

Feel free to fork, star, or contribute to this project!

---

> 📌 **Note**: To explore model training, comparison, and SHAP interpretability in detail, refer to the [Google Colab Notebook](https://colab.research.google.com/drive/1mXZF2xIGM7iZeQA4Ue-uCqTGekTOr1ej?usp=sharing).
