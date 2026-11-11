import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
st.set_page_config(
    page_title="EconoCausal | Coupon Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# THEME
# -----------------------------
if "ec_theme" not in st.session_state:
    st.session_state.ec_theme = "Dark"

with st.sidebar:
    st.markdown("<div class='brand-mark'>EC</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-name'>Econo<span>Causal</span></div>", unsafe_allow_html=True)
    st.caption("CAUSAL INTELLIGENCE · COUPON ANALYTICS")
    st.divider()
    st.markdown("#### WORKSPACE")
    section = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Cleaning and EDA",
            "Double Machine Learning",
            "Uplift Modelling",
            "Budget Optimization",
            "Business Recommendations"
        ],
        format_func=lambda x: {
            "Overview": "⌂   Overview",
            "Data Cleaning and EDA": "▦   Data & EDA",
            "Double Machine Learning": "◈   Double Machine Learning",
            "Uplift Modelling": "⌁   Uplift Modelling",
            "Budget Optimization": "◎   Budget Optimization",
            "Business Recommendations": "✧   Recommendations"
        }[x],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("#### APPEARANCE")
    theme = st.selectbox("Theme", ["Dark", "Light"],
                         index=0 if st.session_state.ec_theme == "Dark" else 1)
    st.session_state.ec_theme = theme
    st.divider()
    st.markdown("#### DATA SOURCE")
    uploaded_file = st.file_uploader(
        "Coupon Impact dataset",
        type=["csv"],
        help="Upload your cleaned Coupon_Impact_Clean.csv file to explore the dataset."
    )
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded")
        except Exception as e:
            st.error(f"Dataset loading failed: {e}")
    else:
        st.caption("No dataset uploaded. Saved research results remain available.")
    st.divider()
    st.caption("EconoCausal · Enterprise Research Workspace")

# -----------------------------
# STYLING
# -----------------------------
if theme == "Dark":
    # Enterprise fintech palette: deep navy + teal
    bg = "#071522"
    panel = "#0D2030"
    panel2 = "#122B3D"
    text = "#EAF3F7"
    muted = "#91A9B8"
    border = "#1E3B4C"
    accent = "#19B6A2"
    accent2 = "#63D6C5"
else:
    bg = "#F2F7F8"
    panel = "#FFFFFF"
    panel2 = "#EAF3F4"
    text = "#102A36"
    muted = "#607985"
    border = "#D8E5E8"
    accent = "#087F79"
    accent2 = "#0C9E91"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {{ color-scheme: {'dark' if theme == 'Dark' else 'light'}; }}
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: {bg}; color: {text}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
    [data-testid="stSidebar"] * {{ color: {text}; }}
    .block-container {{ padding-top: 2.0rem; padding-bottom: 2.5rem; max-width: 1450px; }}
    h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; color: {text} !important; letter-spacing: -0.035em; }}
    p, label, li, [data-testid="stMarkdownContainer"] {{ color: {text}; }}
    [data-testid="stCaptionContainer"] {{ color: {muted}; }}
    .brand-mark {{ width: 42px; height: 42px; border-radius: 13px; display:flex; align-items:center; justify-content:center; background: linear-gradient(135deg, {accent}, #35C6B0); color:white; font-weight:800; font-family:Manrope; font-size:17px; margin-bottom:10px; }}
    .brand-name {{ font-family:Manrope; font-size:23px; font-weight:800; color:{text}; line-height:1.1; }}
    .brand-name span {{ color:{accent}; }}
    .hero {{ background: linear-gradient(112deg, {panel} 0%, {panel2} 68%, #123D49 150%); border:1px solid {border}; border-radius:20px; padding:34px 36px; margin: 0 0 24px 0; position:relative; overflow:hidden; box-shadow:0 16px 42px rgba(0,0,0,.12); }}
    .hero:after {{ content:''; position:absolute; width:330px; height:330px; right:-105px; top:-170px; border-radius:50%; background: radial-gradient(circle, {accent}45, transparent 68%); }}
    .eyebrow {{ color:{accent2}; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:10px; }}
    .hero-title {{ font-family:Manrope; color:{text}; font-size:36px; font-weight:800; line-height:1.12; margin:0 0 12px 0; max-width:780px; white-space:pre-line; }}
    .hero-copy {{ color:{muted}; font-size:14px; max-width:760px; margin:0; }}
    .section-kicker {{ color:{accent2}; font-size:11px; letter-spacing:.14em; font-weight:700; text-transform:uppercase; margin:24px 0 7px 0; }}
    .metric-card {{ background:linear-gradient(145deg, {panel}, {panel2}); border:1px solid {border}; border-top:3px solid {accent}; border-radius:15px; padding:19px 20px; min-height:124px; box-shadow:0 8px 24px rgba(2,20,30,.08); }}
    .metric-label {{ color:{muted}; font-size:12px; font-weight:600; margin-bottom:10px; }}
    .metric-value {{ color:{text}; font-family:Manrope; font-size:27px; font-weight:800; line-height:1.15; }}
    .metric-note {{ color:{muted}; font-size:11px; margin-top:7px; }}
    .info-card {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:20px 21px; height:100%; box-shadow:0 7px 22px rgba(2,20,30,.06); }}
    .info-title {{ color:{text}; font-weight:700; font-size:15px; margin-bottom:7px; }}
    .info-copy {{ color:{muted}; font-size:13px; line-height:1.55; }}
    .step {{ display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-bottom:1px solid {border}; }}
    .step:last-child {{ border-bottom:0; }}
    .step-num {{ background:{panel2}; border:1px solid {border}; color:{accent}; border-radius:10px; min-width:29px; height:29px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; }}
    .step-text {{ color:{text}; font-size:13px; padding-top:5px; }}
    .callout {{ background:{panel2}; border:1px solid {border}; border-left:4px solid {accent}; border-radius:11px; padding:15px 18px; color:{muted}; font-size:13px; }}
    div[data-testid="stDataFrame"] {{ border:1px solid {border}; border-radius:13px; overflow:hidden; }}
    div[data-testid="stMetric"] {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:14px 16px; }}
    div[data-testid="stMetricLabel"] {{ color:{muted}; }}
    div[data-testid="stMetricValue"] {{ color:{text}; font-family:Manrope; }}
    .stRadio label {{ font-size:13px; }}
    .stSelectbox, .stFileUploader {{ color:{text}; }}
    hr {{ border-color:{border}; }}
    .footer {{ color:{muted}; font-size:11px; text-align:center; padding:25px 0 7px 0; letter-spacing:.04em; }}
    [data-testid="stSidebar"] {{ box-shadow: 8px 0 30px rgba(0,0,0,.06); }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:8px 10px; border-radius:9px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:{panel2}; }}
    div[data-testid="stButton"] > button {{ border-radius:10px; border:1px solid {border}; font-weight:700; min-height:42px; }}
    div[data-testid="stButton"] > button[kind="primary"] {{ background:{accent}; border-color:{accent}; color:#FFFFFF; }}
    div[data-testid="stSelectbox"] > div > div, div[data-testid="stFileUploader"] section {{ border-radius:11px; }}
    div[data-testid="stDataFrame"] {{ box-shadow:0 8px 22px rgba(2,20,30,.05); }}
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:8px; }}
    @media (max-width: 900px) {{ .hero {{ padding:24px 22px; }} .hero-title {{ font-size:28px; }} .metric-value {{ font-size:23px; }} }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SAVED COUPON RESEARCH RESULTS
# Values below are retained from the original application.
# -----------------------------
dml_results = pd.DataFrame({
    "Comparison": [
        "High-tier - Amount", "High-tier - Purchase",
        "Low-tier - Amount", "Low-tier - Purchase"
    ],
    "ATE": [-0.2110, -0.0084, -0.0778, -0.0036],
    "P-value": [0.5000, 0.3374, 0.3187, 0.2279]
})

uplift_results = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Average Predicted Uplift": [0.01018, -0.00574, 0.01162]
})

segmentation = pd.DataFrame({
    "Strategy": [
        "Low-tier", "Low-tier", "Low-tier",
        "High-tier T1", "High-tier T1", "High-tier T1",
        "High-tier T2", "High-tier T2", "High-tier T2"
    ],
    "Segment": [
        "Negative Uplift", "Low/Moderate Uplift", "High Uplift",
        "Negative Uplift", "Low/Moderate Uplift", "High Uplift",
        "Negative Uplift", "Low/Moderate Uplift", "High Uplift"
    ],
    "Customers": [3, 1506, 527, 663, 101, 55, 842, 71, 362]
})

budget_results = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Customers Targeted": [1000, 156, 433],
    "Budget Used": [1000, 156, 433],
    "Expected Incremental Purchases": [20.515836, 2.886234, 17.787456],
    "Purchases per Budget Unit": [0.020516, 0.018501, 0.041080]
})

# -----------------------------
# REUSABLE UI HELPERS
# -----------------------------
def hero(title, subtitle, eyebrow="ECONOCAUSAL ANALYTICS"):
    st.markdown(f"""
    <div class='hero'>
      <div class='eyebrow'>{eyebrow}</div>
      <div class='hero-title'>{title}</div>
      <p class='hero-copy'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-note'>{note}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title, description=None):
    st.markdown(f"<div class='section-kicker'>{title}</div>", unsafe_allow_html=True)
    if description:
        st.caption(description)

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
        "features"
    ]

    if not all(key in st.session_state for key in required):
        st.warning(
            "Please configure your dataset in Dataset Configuration first."
        )
        return None

    treatment_column = st.session_state["treatment_column"]
    outcome_column = st.session_state["outcome_column"]
    control_value = st.session_state["control_value"]
    treatment_value = st.session_state["treatment_value"]
    features = st.session_state["features"]

    if not features:
        st.error("Select at least one pre-treatment control variable.")
        return None

    data = df[
        df[treatment_column].isin([control_value, treatment_value])
    ].copy()

    data["__treatment__"] = (
        data[treatment_column] == treatment_value
    ).astype(int)

    data["__outcome__"] = pd.to_numeric(
        data[outcome_column],
        errors="coerce"
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["__treatment__", "__outcome__"])

    if data["__treatment__"].nunique() != 2:
        st.error("Both treatment and control groups are required.")
        return None

    if data["__treatment__"].value_counts().min() < 10:
        st.error("Each group needs at least 10 usable observations.")
        return None

    if data["__outcome__"].nunique() < 2:
        st.error("Outcome must contain at least two different values.")
        return None

    X = prepare_features(data, features)

    if X.shape[1] == 0 or len(X) < 50:
        st.error("Insufficient usable data or features.")
        return None

    return data, X


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        f"{len(df):,}" if df is not None else "Not uploaded"
    )

    col2.metric(
        "Dataset Features",
        len(df.columns) if df is not None else 0
    )

    col3.metric("Supported Formats", "CSV and Excel")

    st.divider()

    st.subheader("Project Objectives")

    st.write("""
    - Analyse different compatible datasets.
    - Identify treatment and outcome variables.
    - Estimate treatment effects using Double Machine Learning.
    - Identify observations with positive predicted uplift.
    - Explore budget allocation opportunities.
    """)

    st.subheader("Project Workflow")

    workflow = [
        "Upload dataset",
        "Data cleaning and exploratory analysis",
        "Configure treatment and outcome variables",
        "Run Double Machine Learning",
        "Perform uplift modelling",
        "Optimize budget",
        "Generate recommendations"
    ]

    for i, item in enumerate(workflow, 1):
        st.markdown(f"**{i}.** {item}")


# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

elif section == "Executive Summary":

    st.header("Executive Summary")
    st.write(
        "A single view of the uploaded dataset and the latest results "
        "from DML, uplift modelling and budget optimization."
    )

    # Dataset-level metrics
    total_customers = len(df) if df is not None else 0

    treatment_groups = "Not configured"
    if (
        df is not None
        and "treatment_column" in st.session_state
        and st.session_state["treatment_column"] in df.columns
    ):
        treatment_groups = int(
            df[st.session_state["treatment_column"]].nunique(dropna=True)
        )

    dml_summary = st.session_state.get("dml_summary")
    uplift_summary = st.session_state.get("uplift_summary")
    budget_summary = st.session_state.get("budget_summary")

    row1 = st.columns(2)
    row1[0].metric(
        "Total Customers / Records",
        f"{total_customers:,}" if df is not None else "Upload dataset"
    )
    row1[1].metric("Treatment Groups", treatment_groups)

    row2 = st.columns(3)
    row2[0].metric(
        "Estimated Treatment Effect (ATE)",
        f"{dml_summary['ate']:.5f}" if dml_summary else "Run DML"
    )
    row2[1].metric(
        "Positive Uplift",
        f"{uplift_summary['positive_count']:,}" if uplift_summary else "Run Uplift"
    )
    row2[2].metric(
        "Budget Efficiency",
        f"{budget_summary['efficiency']:.5f}" if budget_summary else "Run Budget Optimization"
    )

    st.divider()
    st.subheader("Latest Analysis Details")

    if dml_summary:
        st.markdown(
            f"- **DML p-value:** {dml_summary['p_value']:.5f}\n"
            f"- **95% confidence interval:** "
            f"{dml_summary['ci_lower']:.5f} to {dml_summary['ci_upper']:.5f}"
        )
        if dml_summary["p_value"] < 0.05:
            st.success("The latest DML estimate is statistically significant at the 5% level.")
        else:
            st.info("The latest DML estimate is not statistically significant at the 5% level.")
    else:
        st.caption("DML results will appear here after you run Double Machine Learning.")

    if uplift_summary:
        st.markdown(
            f"- **Average predicted uplift:** {uplift_summary['average_uplift']:.5f}\n"
            f"- **Negative uplift observations:** {uplift_summary['negative_count']:,}"
        )
    else:
        st.caption("Uplift results will appear here after you run Uplift Modelling.")

    if budget_summary:
        st.markdown(
            f"- **Customers targeted:** {budget_summary['customers_targeted']:,}\n"
            f"- **Budget used:** {budget_summary['budget_used']:.2f}\n"
            f"- **Expected incremental purchases (model estimate):** "
            f"{budget_summary['expected_incremental_purchases']:.3f}"
        )
    else:
        st.caption("Budget results will appear here after you run Budget Optimization.")

    st.warning(
        "These figures summarize the latest analyses in this app session. "
        "Predicted uplift and budget efficiency are model-based estimates, "
        "not verified individual causal effects or guaranteed business returns."
    )


# --------------------------------------------------
# DATA CLEANING AND EDA
# --------------------------------------------------

elif section == "Data Cleaning and EDA":

    st.header("Data Cleaning and Exploratory Data Analysis")

    if df is None:
        st.warning("Please upload a CSV or Excel dataset.")

    else:
        st.subheader("Dataset Preview")

        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])

        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Quality")

        quality = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Missing Values",
            int(df.isnull().sum().sum())
        )

        col2.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        st.subheader("Numerical Summary")
        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            selected_column = st.selectbox(
                "Choose a numerical column",
                numeric_columns
            )

            st.bar_chart(
                df[selected_column].value_counts()
                .sort_index()
                .head(30)
            )

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if categorical_columns:
            selected_category = st.selectbox(
                "Choose a categorical column",
                categorical_columns
            )

            st.bar_chart(
                df[selected_category].value_counts().head(20)
            )


# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

elif section == "Dataset Configuration":

    st.header("Dataset Configuration")

    if df is None:
        st.warning("Upload a dataset first.")

    else:
        columns = df.columns.tolist()

        treatment_column = st.selectbox(
            "Treatment Column",
            columns
        )

        treatment_values = (
            df[treatment_column].dropna().unique().tolist()
        )

        if len(treatment_values) >= 2:

            control_value = st.selectbox(
                "Control Group Value",
                treatment_values
            )

            treatment_options = [
                value for value in treatment_values
                if value != control_value
            ]

            treatment_value = st.selectbox(
                "Treatment Group Value",
                treatment_options
            )

            outcome_options = [
                col for col in columns
                if col != treatment_column
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if outcome_options:

                outcome_column = st.selectbox(
                    "Outcome Column",
                    outcome_options
                )

                feature_options = [
                    col for col in columns
                    if col not in [treatment_column, outcome_column]
                ]

                selected_features = st.multiselect(
                    "Pre-treatment Control Variables",
                    feature_options
                )

                st.warning(
                    "Select only variables measured before treatment. "
                    "Avoid IDs, leakage variables and post-treatment variables."
                )

                if st.button("Save Configuration"):

                    st.session_state["treatment_column"] = treatment_column
                    st.session_state["control_value"] = control_value
                    st.session_state["treatment_value"] = treatment_value
                    st.session_state["outcome_column"] = outcome_column
                    st.session_state["features"] = selected_features

                    st.success("Configuration saved successfully.")

                    st.write("Treatment:", treatment_column)
                    st.write("Outcome:", outcome_column)
                    st.write("Control variables:", selected_features)

            else:
                st.error("No numerical outcome column is available.")

        else:
            st.error("Treatment column must contain at least two values.")


# --------------------------------------------------
# DOUBLE MACHINE LEARNING
# --------------------------------------------------

elif section == "Double Machine Learning":

    st.header("Double Machine Learning")

    st.write(
        "Estimate the average treatment effect using "
        "Double Machine Learning with cross-fitting."
    )

    if st.button("Run DML Analysis"):

        prepared = get_analysis_data()

        if prepared is not None:

            data, X = prepared

            try:
                from doubleml import DoubleMLData, DoubleMLPLR

                y = data["__outcome__"].to_numpy(dtype=float)
                d = data["__treatment__"].to_numpy(dtype=float)

                ml_l = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )

                ml_m = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )

                dml_data = DoubleMLData.from_arrays(
                    x=X.to_numpy(),
                    y=y,
                    d=d
                )

                model = DoubleMLPLR(
                    dml_data,
                    ml_l=ml_l,
                    ml_m=ml_m,
                    n_folds=5
                )

                with st.spinner("Training DML model..."):
                    model.fit()

                result = pd.DataFrame({
                    "ATE": [model.coef[0]],
                    "Standard Error": [model.se[0]],
                    "P-value": [model.pval[0]]
                })

                lower, upper = model.confint().iloc[0].tolist()
                st.session_state["dml_summary"] = {
                    "ate": float(model.coef[0]),
                    "standard_error": float(model.se[0]),
                    "p_value": float(model.pval[0]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper)
                }

                st.subheader("DML Results")

                st.dataframe(
                    result.style.format("{:.5f}"),
                    use_container_width=True
                )

                st.write(
                    f"95% Confidence Interval: "
                    f"{lower:.5f} to {upper:.5f}"
                )

                if model.pval[0] < 0.05:
                    st.success(
                        "The estimated effect is statistically significant "
                        "at the 5% level."
                    )
                else:
                    st.info(
                        "The analysis did not find statistically significant "
                        "evidence of a treatment effect at the 5% level."
                    )

                st.warning(
                    "Causal interpretation depends on treatment assignment, "
                    "pre-treatment controls and model assumptions."
                )

            except Exception as e:
                st.error(f"DML analysis failed: {e}")


# --------------------------------------------------
# UPLIFT MODELLING
# --------------------------------------------------

elif section == "Uplift Modelling":

    st.header("Uplift Modelling")

    st.write(
        "T-Learner compares predicted outcomes under treatment "
        "and control conditions."
    )

    if st.button("Run Uplift Analysis"):

        prepared = get_analysis_data()

        if prepared is not None:

            data, X = prepared

            y = data["__outcome__"]
            d = data["__treatment__"]

            if not y.isin([0, 1]).all():

                st.error(
                    "This T-Learner implementation requires a binary "
                    "outcome containing only 0 and 1."
                )

            elif (
                y[d == 0].nunique() < 2
                or y[d == 1].nunique() < 2
            ):

                st.error(
                    "Both treatment and control groups must contain "
                    "both outcome classes (0 and 1)."
                )

            else:

                try:
                    X_train, X_test, y_train, y_test, d_train, d_test = (
                        train_test_split(
                            X,
                            y,
                            d,
                            test_size=0.30,
                            random_state=42,
                            stratify=d
                        )
                    )

                    control_model = LogisticRegression(max_iter=1000)
                    treatment_model = LogisticRegression(max_iter=1000)

                    control_model.fit(
                        X_train[d_train == 0],
                        y_train[d_train == 0]
                    )

                    treatment_model.fit(
                        X_train[d_train == 1],
                        y_train[d_train == 1]
                    )

                    p_control = control_model.predict_proba(
                        X_test
                    )[:, 1]

                    p_treatment = treatment_model.predict_proba(
                        X_test
                    )[:, 1]

                    uplift = p_treatment - p_control

                    uplift_results = pd.DataFrame({
                        "Predicted Control Probability": p_control,
                        "Predicted Treatment Probability": p_treatment,
                        "Predicted Uplift": uplift
                    })

                    st.session_state["uplift_results"] = uplift_results
                    st.session_state["uplift_summary"] = {
                        "average_uplift": float(np.mean(uplift)),
                        "positive_count": int((uplift > 0).sum()),
                        "negative_count": int((uplift < 0).sum())
                    }

                    st.subheader("Uplift Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Average Predicted Uplift",
                        f"{np.mean(uplift):.5f}"
                    )

                    col2.metric(
                        "Positive Uplift",
                        int((uplift > 0).sum())
                    )

                    col3.metric(
                        "Negative Uplift",
                        int((uplift < 0).sum())
                    )

                    st.dataframe(
                        uplift_results.head(20),
                        use_container_width=True
                    )

                    st.subheader("Predicted Uplift Distribution")

                    st.bar_chart(
                        uplift_results["Predicted Uplift"]
                        .value_counts(bins=20)
                        .sort_index()
                    )

                    st.info(
                        "These are model predictions, not automatically "
                        "validated individual causal effects."
                    )

                except Exception as e:
                    st.error(f"Uplift modelling failed: {e}")


# --------------------------------------------------
# BUDGET OPTIMIZATION
# --------------------------------------------------

elif section == "Budget Optimization":

    st.header("Budget Optimization")

    st.write(
        "Allocate a limited budget to observations with "
        "the highest positive predicted uplift."
    )

    uplift_results = st.session_state.get("uplift_results")

    if uplift_results is None:

        st.warning("Run Uplift Modelling first.")

    else:

        budget = st.number_input(
            "Available Budget",
            min_value=1.0,
            value=1000.0
        )

        coupon_cost = st.number_input(
            "Cost per Treatment",
            min_value=0.01,
            value=1.0
        )

        if st.button("Optimize Budget"):

            max_customers = int(budget // coupon_cost)

            positive_customers = (
                uplift_results[
                    uplift_results["Predicted Uplift"] > 0
                ]
                .sort_values(
                    "Predicted Uplift",
                    ascending=False
                )
            )

            selected = positive_customers.head(max_customers)

            customers_targeted = len(selected)

            expected_incremental_purchases = (
                selected["Predicted Uplift"].sum()
            )

            budget_used = customers_targeted * coupon_cost

            efficiency = (
                expected_incremental_purchases / budget_used
                if budget_used > 0 else 0
            )

            st.session_state["budget_summary"] = {
                "customers_targeted": int(customers_targeted),
                "budget_used": float(budget_used),
                "expected_incremental_purchases": float(expected_incremental_purchases),
                "efficiency": float(efficiency)
            }

            st.subheader("Budget Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Customers Targeted", customers_targeted)
            col2.metric("Budget Used", f"{budget_used:.2f}")

            col3.metric(
                "Expected Incremental Purchases",
                f"{expected_incremental_purchases:.3f}"
            )

            st.metric(
                "Estimated Purchases per Budget Unit",
                f"{efficiency:.5f}"
            )

            st.dataframe(selected, use_container_width=True)

            st.warning(
                "Budget calculations are model-based estimates. "
                "They do not represent verified profit or savings."
            )


# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

elif section == "Business Recommendations":

    st.header("Business Recommendations")

    st.subheader("General Recommendations")

    recommendations = [
        "Use suitable pre-treatment variables for causal analysis.",
        "Verify how treatment groups were assigned.",
        "Compare model predictions with real business outcomes.",
        "Validate uplift predictions before targeting customers.",
        "Use actual treatment costs for budget allocation."
    ]

    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {recommendation}")

    st.subheader("Project Limitations")

    limitations = [
        "Not every dataset is suitable for causal analysis.",
        "Treatment and outcome variables must be correctly defined.",
        "Post-treatment variables should not be used as ordinary controls.",
        "Model predictions depend on data quality and assumptions.",
        "Further validation is needed before business deployment."
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.success(
        "EconoCausal provides a configurable workflow for "
        "exploring treatment effects, uplift and budget allocation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EconoCausal | Multi-Dataset Causal Analysis Platform"
)

st.caption(
    "Research application | Results require appropriate "
    "statistical and business validation."
)
