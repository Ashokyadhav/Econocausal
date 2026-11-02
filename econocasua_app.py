import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EconoCausal")
st.subheader("Dynamic Pricing via Double Machine Learning")

st.info(
    "Research dashboard showing existing cluster-aware "
    "Double Machine Learning results."
)

# Cluster-aware model results
baseline = {
    "coefficient": -97.0281,
    "se": 10.2362,
    "lower": -117.0907,
    "upper": -76.9655,
    "observations": 20000
}

filtered = {
    "coefficient": -27.3148,
    "se": 0.9820,
    "lower": -29.2395,
    "upper": -25.3901,
    "observations": 19572
}

# Main metrics
st.header("Key Results")

col1, col2, col3 = st.columns(3)

col1.metric("Baseline Effect", f"{baseline['coefficient']:.2f}")
col2.metric("Filtered Effect", f"{filtered['coefficient']:.2f}")
col3.metric("Filtered Standard Error", f"{filtered['se']:.3f}")

# Comparison table
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
        baseline["se"],
        baseline["lower"],
        baseline["upper"]
    ],
    "Filtered": [
        filtered["observations"],
        filtered["coefficient"],
        filtered["se"],
        filtered["lower"],
        filtered["upper"]
    ]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

# Effect chart
st.header("Estimated Price Effect")

effect_df = pd.DataFrame({
    "Model": ["Baseline", "Filtered"],
    "Estimated Effect": [
        baseline["coefficient"],
        filtered["coefficient"]
    ]
})

st.bar_chart(effect_df.set_index("Model"))

# Standard error chart
st.header("Standard Error Comparison")

se_df = pd.DataFrame({
    "Model": ["Baseline", "Filtered"],
    "Standard Error": [
        baseline["se"],
        filtered["se"]
    ]
})

st.bar_chart(se_df.set_index("Model"))

# Confidence intervals
st.header("Confidence Intervals")

ci_df = pd.DataFrame({
    "Model": ["Baseline", "Filtered"],
    "Lower Limit": [baseline["lower"], filtered["lower"]],
    "Upper Limit": [baseline["upper"], filtered["upper"]]
})

st.dataframe(ci_df, use_container_width=True, hide_index=True)

# Project information
st.header("Project Details")

details = pd.DataFrame({
    "Item": [
        "Dataset",
        "Treatment",
        "Outcome",
        "Control Variables",
        "Method",
        "Inference",
        "Cluster"
    ],
    "Details": [
        "Cleaned WCER2",
        "d_PRICE",
        "d_MOVE",
        "STORE, UPC, WEEK",
        "DoubleMLPLR",
        "Cluster-aware",
        "PAIR_ID"
    ]
})

st.dataframe(details, use_container_width=True, hide_index=True)

# Interpretation
st.header("Research Interpretation")

st.write(
    "Both models estimate a negative relationship between "
    "price changes and sales movement changes."
)

st.write(
    "The filtered model estimates a smaller negative relationship "
    "than the baseline model. This shows that the estimated effect "
    "is sensitive to the filtering procedure."
)

st.warning(
    "These estimates come from different samples. Further "
    "robustness checks are needed before making strong causal claims."
)

st.divider()

st.caption(
    "EconoCausal | Dynamic Pricing via Double Machine Learning"
)
