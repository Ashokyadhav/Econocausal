import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EconoCausal")
st.subheader("Dynamic Pricing and Coupon Effectiveness Analysis")

st.write(
    "An integrated data science dashboard combining "
    "Data Analysis, Double Machine Learning, "
    "Uplift Modelling and Budget Optimization."
)

st.info(
    "This dashboard displays dataset analysis and saved model results. "
    "It does not train machine learning models."
)

# --------------------------------------------------
# SAVED DML RESULTS
# --------------------------------------------------

dml_results = pd.DataFrame({
    "Comparison": [
        "High-tier - Amount",
        "High-tier - Purchase",
        "Low-tier - Amount",
        "Low-tier - Purchase"
    ],
    "ATE": [-0.2110, -0.0084, -0.0778, -0.0036],
    "P-value": [0.5000, 0.3374, 0.3187, 0.2279]
})

# --------------------------------------------------
# SAVED UPLIFT RESULTS
# --------------------------------------------------

uplift_results = pd.DataFrame({
    "Strategy": [
        "Low-tier",
        "High-tier T1",
        "High-tier T2"
    ],
    "Average Predicted Uplift": [
        0.01018,
        -0.00574,
        0.01162
    ]
})

# --------------------------------------------------
# SAVED CUSTOMER SEGMENTATION RESULTS
# --------------------------------------------------

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
    "Customers": [
        3, 1506, 527,
        663, 101, 55,
        842, 71, 362
    ]
})

# --------------------------------------------------
# SAVED BUDGET RESULTS
# --------------------------------------------------

budget_results = pd.DataFrame({
    "Strategy": [
        "Low-tier",
        "High-tier T1",
        "High-tier T2"
    ],
    "Customers Targeted": [1000, 156, 433],
    "Budget Used": [1000, 156, 433],
    "Expected Incremental Purchases": [
        20.515836,
        2.886234,
        17.787456
    ],
    "Purchases per Budget Unit": [
        0.020516,
        0.018501,
        0.041080
    ]
})

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Project Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Data Cleaning and EDA",
        "Double Machine Learning",
        "Uplift Modelling",
        "Budget Optimization",
        "Business Recommendations"
    ]
)

st.sidebar.divider()
st.sidebar.caption("EconoCausal Project")
st.sidebar.caption("Coupon Impact Dataset")
st.sidebar.caption("Double Machine Learning")

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.sidebar.subheader("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Coupon_Impact_Clean.csv",
    type=["csv"]
)

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.sidebar.success("Dataset loaded successfully")

    except Exception as e:
        st.sidebar.error(f"Dataset loading failed: {e}")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "12,959")
    col2.metric("Dataset Features", "8")
    col3.metric("Project Modules", "3")

    st.divider()

    st.subheader("Project Objectives")

    st.write("""
    - Analyse customer purchasing behaviour.
    - Estimate coupon treatment effects using DML.
    - Identify customers with positive predicted uplift.
    - Compare coupon strategies.
    - Improve marketing budget allocation.
    """)

    st.subheader("Project Workflow")

    workflow = [
        "Data Cleaning and Exploratory Data Analysis",
        "Treatment and Control Group Analysis",
        "Double Machine Learning",
        "Uplift Modelling using T-Learner",
        "Customer Segmentation",
        "Budget Optimization",
        "Business Recommendations"
    ]

    for i, step in enumerate(workflow, 1):
        st.markdown(f"**{i}.** {step}")

    st.divider()

    st.subheader("Technology Stack")

    st.write(
        "Python | Pandas | NumPy | Matplotlib | "
        "Scikit-learn | DoubleML | Streamlit | GitHub"
    )

# --------------------------------------------------
# DATA CLEANING AND EDA
# --------------------------------------------------

elif section == "Data Cleaning and EDA":

    st.header("Data Cleaning and Exploratory Data Analysis")

    if df is None:
        st.warning(
            "Upload Coupon_Impact_Clean.csv using the sidebar "
            "to view dataset analysis."
        )

    else:

        st.subheader("Dataset Preview")

        col1, col2 = st.columns(2)

        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("Data Quality")

        quality = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        col2.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        if "group" in df.columns:

            st.subheader("Experimental Group Distribution")

            group_counts = df["group"].value_counts()

            st.bar_chart(group_counts)

            st.subheader("Purchase Rate by Group")

            purchase_rates = (
                df.groupby("group")["purchase"]
                .mean()
                .mul(100)
                .rename("Purchase Rate (%)")
            )

            st.bar_chart(purchase_rates)

            st.dataframe(
                purchase_rates.reset_index(),
                use_container_width=True,
                hide_index=True
            )

            if "amount" in df.columns:

                st.subheader("Average Purchase Amount")

                average_amount = (
                    df.groupby("group")["amount"]
                    .mean()
                    .rename("Average Amount")
                )

                st.bar_chart(average_amount)

# --------------------------------------------------
# DOUBLE MACHINE LEARNING
# --------------------------------------------------

elif section == "Double Machine Learning":

    st.header("Double Machine Learning Results")

    st.write("""
    Double Machine Learning estimates the average effect
    of coupon treatment on customer purchasing behaviour
    while accounting for relevant observed factors.
    """)

    st.subheader("Model Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Estimator",
            "Outcome Model",
            "Treatment Model",
            "Cross Validation",
            "Outcome Variables"
        ],
        "Value": [
            "DoubleMLPLR",
            "Random Forest Regressor",
            "Logistic Regression",
            "5 Folds",
            "Purchase and Amount"
        ]
    })

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("ATE Results")

    st.dataframe(
        dml_results.style.format({
            "ATE": "{:.4f}",
            "P-value": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Average Treatment Effect Comparison")

    st.bar_chart(
        dml_results.set_index("Comparison")["ATE"]
    )

    st.subheader("Statistical Interpretation")

    st.write("""
    All four comparisons have p-values greater than 0.05.

    Therefore, these analyses did not find statistically
    significant evidence of a coupon treatment effect.

    This does not prove that coupons have no effect.
    """)

    st.warning(
        "Causal interpretation depends on valid treatment assignment, "
        "appropriate pre-treatment controls and model assumptions."
    )

# --------------------------------------------------
# UPLIFT MODELLING
# --------------------------------------------------

elif section == "Uplift Modelling":

    st.header("Uplift Modelling")

    st.write("""
    Uplift modelling estimates how customer purchase probability
    may change when a customer receives a coupon.
    """)

    st.subheader("T-Learner Method")

    st.write("""
    T-Learner uses two separate models:

    1. Control Model: Predicts purchase probability without treatment.
    2. Treatment Model: Predicts purchase probability with treatment.

    Uplift Score = Treatment Prediction - Control Prediction
    """)

    st.subheader("Strategy Comparisons")

    st.dataframe(
        uplift_results.style.format({
            "Average Predicted Uplift": "{:.5f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Average Predicted Uplift")

    st.bar_chart(
        uplift_results.set_index("Strategy")
    )

    st.subheader("Customer Segmentation")

    st.dataframe(
        segmentation,
        use_container_width=True,
        hide_index=True
    )

    segment_chart = segmentation.pivot(
        index="Strategy",
        columns="Segment",
        values="Customers"
    )

    st.subheader("Customer Segment Distribution")

    st.bar_chart(segment_chart)

    st.info(
        "Uplift scores are model predictions. "
        "Their causal interpretation requires validation."
    )

# --------------------------------------------------
# BUDGET OPTIMIZATION
# --------------------------------------------------

elif section == "Budget Optimization":

    st.header("Budget Optimization")

    st.write("""
    This module ranks customers using predicted uplift scores
    and compares coupon strategies under an illustrative
    budget scenario.
    """)

    st.subheader("Budget Results")

    st.dataframe(
        budget_results.style.format({
            "Expected Incremental Purchases": "{:.2f}",
            "Purchases per Budget Unit": "{:.5f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Expected Incremental Purchases")

    st.bar_chart(
        budget_results.set_index("Strategy")[
            "Expected Incremental Purchases"
        ]
    )

    st.subheader("Estimated Budget Efficiency")

    st.bar_chart(
        budget_results.set_index("Strategy")[
            "Purchases per Budget Unit"
        ]
    )

    best_strategy = budget_results.loc[
        budget_results["Purchases per Budget Unit"].idxmax()
    ]

    highest_volume = budget_results.loc[
        budget_results["Expected Incremental Purchases"].idxmax()
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Estimated Budget Efficiency",
        best_strategy["Strategy"]
    )

    col2.metric(
        "Highest Estimated Purchase Volume",
        highest_volume["Strategy"]
    )

    st.warning(
        "Budget and coupon costs are illustrative assumptions. "
        "These results are not actual profit or savings estimates."
    )

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

elif section == "Business Recommendations":

    st.header("Business Recommendations")

    st.subheader("Key Findings")

    st.write("""
    1. High-tier T2 has the highest average predicted uplift.

    2. High-tier T2 has the highest estimated purchases
       per budget unit in the illustrative scenario.

    3. Low-tier has the highest estimated incremental
       purchase volume in the budget scenario.

    4. DML results did not show statistically significant
       treatment effects in the four comparisons.
    """)

    st.subheader("Recommendations")

    recommendations = [
        "Consider uplift-based customer targeting.",
        "Prioritize customers with positive predicted uplift.",
        "Compare coupon strategies using actual coupon costs.",
        "Validate model predictions through controlled experiments.",
        "Reassess budget allocation using real business outcomes."
    ]

    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {recommendation}")

    st.subheader("Project Limitations")

    limitations = [
        "Treatment assignment and experiment design need verification.",
        "Pre-treatment timing of predictor variables must be confirmed.",
        "Uplift predictions are not automatically causal effects.",
        "Budget values and coupon costs are illustrative.",
        "Further validation is required before business deployment."
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.success(
        "EconoCausal combines data analysis, causal machine learning "
        "and uplift modelling to support informed coupon allocation."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EconoCausal | Dynamic Pricing and Coupon Effectiveness Analysis"
)

st.caption(
    "Research dashboard | Model results and budget scenarios "
    "are subject to validation."
)
