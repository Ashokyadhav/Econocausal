import streamlit as st
import pandas as pd

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="EconoCausal | Dynamic Pricing",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# MODEL RESULTS
# Existing cluster-aware results
# -----------------------------------
baseline = {
    "name": "Baseline",
    "observations": 20000,
    "coefficient": -97.0281,
    "std_error": 10.2362,
    "lower": -117.0907,
    "upper": -76.9655
}

filtered = {
    "name": "Filtered",
    "observations": 19572,
    "coefficient": -27.3148,
    "std_error": 0.9820,
    "lower": -29.2395,
    "upper": -25.3901
}

# -----------------------------------
# HEADER
# -----------------------------------
st.title("📊 EconoCausal")
st.subheader("Dynamic Pricing via Double Machine Learning")

st.markdown(
    "An analytical dashboard exploring the relationship "
    "between price changes and sales movement changes."
)

st.info(
    "This dashboard displays existing cluster-aware DML results. "
    "It does not train or rerun any machine learning models."
)

st.divider()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("Project Navigation")
section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Comparison",
        "Research Interpretation",
        "Methodology"
    ]
)

st.sidebar.divider()
st.sidebar.caption("EconoCausal Project")
st.sidebar.caption("Double Machine Learning")
st.sidebar.caption("Inference: Cluster-aware by PAIR_ID")

# -----------------------------------
# OVERVIEW
# -----------------------------------
if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baseline Effect",
            f"{baseline['coefficient']:.2f}"
        )

    with col2:
        st.metric(
            "Filtered Effect",
            f"{filtered['coefficient']:.2f}"
        )

    with col3:
        st.metric(
            "Filtered Standard Error",
            f"{filtered['std_error']:.3f}"
        )

    st.divider()

    st.subheader("Dataset and Model Information")

    details = pd.DataFrame({
        "Parameter": [
            "Dataset",
            "Treatment Variable",
            "Outcome Variable",
            "Control Variables",
            "Estimation Method",
            "Inference Method",
            "Cluster Variable"
        ],
        "Value": [
            "Cleaned WCER2",
            "d_PRICE",
            "d_MOVE",
            "STORE, UPC, WEEK",
            "DoubleMLPLR",
            "Cluster-aware",
            "PAIR_ID"
        ]
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Observations Used")

    sample_data = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Observations": [
            baseline["observations"],
            filtered["observations"]
        ]
    })

    st.bar_chart(
        sample_data.set_index("Model")
    )

# -----------------------------------
# MODEL COMPARISON
# -----------------------------------
elif section == "Model Comparison":

    st.header("Model Comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "Observations",
            "Coefficient",
            "Standard Error",
            "95% CI Lower",
            "95% CI Upper"
        ],
        "Baseline": [
            baseline["observations"],
            baseline["coefficient"],
            baseline["std_error"],
            baseline["lower"],
            baseline["upper"]
        ],
        "Filtered": [
            filtered["observations"],
            filtered["coefficient"],
            filtered["std_error"],
            filtered["lower"],
            filtered["upper"]
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Estimated Price Effect")

    effect_data = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Estimated Effect": [
            baseline["coefficient"],
            filtered["coefficient"]
        ]
    })

    st.bar_chart(
        effect_data.set_index("Model")
    )

    st.subheader("Standard Error Comparison")

    error_data = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Standard Error": [
            baseline["std_error"],
            filtered["std_error"]
        ]
    })

    st.bar_chart(
        error_data.set_index("Model")
    )

    st.subheader("Confidence Interval Comparison")

    ci_data = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Lower Limit": [
            baseline["lower"],
            filtered["lower"]
        ],
        "Upper Limit": [
            baseline["upper"],
            filtered["upper"]
        ]
    })

    st.dataframe(
        ci_data,
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------
# RESEARCH INTERPRETATION
# -----------------------------------
elif section == "Research Interpretation":

    st.header("Research Interpretation")

    st.subheader("Main Findings")

    st.write(
        "Both models estimate a negative relationship between "
        "price changes and sales movement changes."
    )

    st.write(
        f"The baseline model estimates a coefficient of "
        f"{baseline['coefficient']:.4f}."
    )

    st.write(
        f"The filtered model estimates a coefficient of "
        f"{filtered['coefficient']:.4f}."
    )

    st.subheader("What Does This Mean?")

    st.write(
        "The negative estimates indicate that higher price changes "
        "are associated with lower sales movement changes, "
        "under the respective model specifications."
    )

    st.warning(
        "The estimates differ substantially. The models use "
        "different samples, so the difference should not be "
        "attributed to filtering alone without further investigation. "
        "Additional robustness checks are needed before making "
        "strong causal claims."
    )

    st.subheader("Confidence Intervals")

    st.write(
        f"Baseline: [{baseline['lower']:.4f}, "
        f"{baseline['upper']:.4f}]"
    )

    st.write(
        f"Filtered: [{filtered['lower']:.4f}, "
        f"{filtered['upper']:.4f}]"
    )

# -----------------------------------
# METHODOLOGY
# -----------------------------------
elif section == "Methodology":

    st.header("Project Methodology")

    st.subheader("1. Dataset")

    st.write(
        "The project uses the cleaned WCER2 dataset containing "
        "store, product, week, price and sales movement information."
    )

    st.subheader("2. Treatment and Outcome")

    st.write(
        "**Treatment (d_PRICE):** Change in price."
    )

    st.write(
        "**Outcome (d_MOVE):** Change in sales movement."
    )

    st.subheader("3. Control Variables")

    st.write("STORE, UPC and WEEK.")

    st.subheader("4. Estimation Method")

    st.write(
        "Double Machine Learning using the Partialling Out "
        "method (DoubleMLPLR)."
    )

    st.subheader("5. Cluster-aware Inference")

    st.write(
        "PAIR_ID is used as the clustering variable to account "
        "for repeated observations within store-product pairs."
    )

    st.info(
        "The dashboard presents saved model results. "
        "It does not perform model training."
    )
