import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EconoCausal - Multi Dataset",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EconoCausal")
st.subheader("Multi-Dataset Causal Analysis Platform")

st.write(
    "Upload compatible datasets, explore the data, "
    "configure treatment and outcome variables, and "
    "perform causal and uplift analysis."
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Project Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Executive Summary",
        "Data Cleaning and EDA",
        "Dataset Configuration",
        "Double Machine Learning",
        "Uplift Modelling",
        "Budget Optimization",
        "Business Recommendations"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel file",
    type=["csv", "xlsx"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.caption(f"File: {uploaded_file.name}")
        st.sidebar.caption(f"Rows: {len(df):,}")
        st.sidebar.caption(f"Columns: {len(df.columns)}")

    except Exception as e:
        st.sidebar.error(f"Unable to load dataset: {e}")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def prepare_features(data, columns):
    X = data[columns].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(float)


def get_analysis_data():

    if df is None:
        st.error("Please upload a dataset.")
        return None

    required = [
        "treatment_column",
        "control_value",
        "treatment_value",
        "outcome_column",
