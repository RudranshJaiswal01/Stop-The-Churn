# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model (replace with your actual model path)
MODEL_PATH = 'model.pkl'
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# Sidebar: Upload CSV
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# UI: Dark mode toggle
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
            .main, .stApp {background-color: #1E1E1E; color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )

# App logic
def predict_and_display(df):
    if 'CustomerID' not in df.columns:
        st.error("Missing 'CustomerID' column.")
        return

    # Predict churn probabilities
    X = df.drop(columns=['CustomerID'], errors='ignore')
    y_probs = model.predict_proba(X)[:, 1]
    df['Churn_Probability'] = y_probs

    # Sort by churn probability
    df_sorted = df.sort_values(by='Churn_Probability', ascending=False)
    top_churn = df_sorted.head(10).copy()

    st.subheader("Top 10 Customers Likely to Churn")

    # Checkbox states
    if 'selected_ids' not in st.session_state:
        st.session_state.selected_ids = set()
        st.session_state.previous_ids = set()

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        show_all = st.checkbox("Show All", key="show_all")
    with col2:
        clear_all = st.checkbox("Clear All", key="clear_all")

    if show_all:
        st.session_state.previous_ids = st.session_state.selected_ids.copy()
        st.session_state.selected_ids = set(top_churn['CustomerID'])
    elif clear_all:
        st.session_state.previous_ids = st.session_state.selected_ids.copy()
        st.session_state.selected_ids = set()
    elif not show_all and not clear_all:
        st.session_state.selected_ids = st.session_state.previous_ids.copy()

    for _, row in top_churn.iterrows():
        checked = st.checkbox(f"Customer {row['CustomerID']}", value=row['CustomerID'] in st.session_state.selected_ids)
        if checked:
            st.session_state.selected_ids.add(row['CustomerID'])
        else:
            st.session_state.selected_ids.discard(row['CustomerID'])

    chart_data = top_churn[top_churn['CustomerID'].isin(st.session_state.selected_ids)]

    if not chart_data.empty:
        def color(prob):
            if prob > 0.7:
                return 'red'
            elif prob > 0.3:
                return 'yellow'
            else:
                return 'green'

        chart_data['Color'] = chart_data['Churn_Probability'].apply(color)
        fig = px.bar(
            chart_data,
            x='CustomerID',
            y='Churn_Probability',
            color='Color',
            color_discrete_map={'red': 'red', 'yellow': 'yellow', 'green': 'green'},
            title="Churn Probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Search & filter
    st.subheader("Search & Filter")
    search_id = st.text_input("Search by CustomerID")
    min_prob = st.slider("Minimum Churn Probability", 0.0, 1.0, 0.0)
    max_prob = st.slider("Maximum Churn Probability", 0.0, 1.0, 1.0)

    filtered = df_sorted[(df_sorted['Churn_Probability'] >= min_prob) & (df_sorted['Churn_Probability'] <= max_prob)]

    if search_id:
        filtered = filtered[filtered['CustomerID'].astype(str).str.contains(search_id)]

    st.dataframe(filtered, use_container_width=True)

# Main
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    predict_and_display(df)
else:
    st.info("Please upload a CSV file to begin.")
