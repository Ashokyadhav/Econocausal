import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide"
)

# Project heading
st.title("EconoCausal")
st.subheader("Dynamic Pricing via Double Machine Learning")

st.info(
    "Research dashboard presenting cluster-aware "
    "Double Machine Learning results."
)

# Validated model results
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

# Main results
st.header("Model Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline Effect", "-97.03")
col2.metric("Filtered Effect", "-27.31")
col3.metric("Baseline SE", "10.2362")
col4.metric("Filtered SE", "0.9820")

# Results table
st.header("Detailed Model Comparison")

results = pd.DataFrame({
    "Metric": [
        "Observations",
        "Coefficient",
        "Standard Error",
        "95% CI Lower",
        "95% CI Upper"
    ],
    "Baseline Model": [
        baseline["observations"],
        baseline["coefficient"],
        baseline["se"],
        baseline["lower"],
        baseline["upper"]
    ],
    "Filtered Model": [
        filtered["observations"],
        filtered["coefficient"],
        filtered["se"],
        filtered["lower"],
        filtered["upper"]
    ]
})

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

# Coefficient visualization
st.header("Price Effect Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

models = ["Baseline", "Filtered"]
coefficients = [
    baseline["coefficient"],
    filtered["coefficient"]
]

lower_errors = [
    baseline["coefficient"] - baseline["lower"],
    filtered["coefficient"] - filtered["lower"]
]

upper_errors = [
    baseline["upper"] - baseline["coefficient"],
    filtered["upper"] - filtered["coefficient"]
]

ax.errorbar(
    models,
    coefficients,
    yerr=[lower_errors, upper_errors],
    fmt="o",
    capsize=8,
    markersize=9
)

ax.axhline(0, linestyle="--")
ax.set_ylabel("Estimated Price Effect on Sales Movement")
ax.set_title("Model Coefficients with 95% Confidence Intervals")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# Standard error visualization
st.header("Standard Error Comparison")

se_data = pd.DataFrame({
    "Model": ["Baseline", "Filtered"],
    "Standard Error": [
        baseline["se"],
        filtered["se"]
    ]
})

st.bar_chart(
    se_data.set_index("Model")
)

# Methodology
st.header("Model Details")

details = pd.DataFrame({
    "Item": [
        "Dataset",
        "Treatment",
        "Outcome",
        "Controls",
        "Estimator",
        "Inference",
        "Clustering"
    ],
    "Details": [
        "Cleaned WCER2",
        "d_PRICE (change in price)",
        "d_MOVE (change in sales movement)",
        "STORE, UPC, WEEK",
        "DoubleMLPLR",
        "Cluster-aware inference",
        "PAIR_ID"
    ]
})

st.dataframe(
    details,
    use_container_width=True,
    hide_index=True
)

# Interpretation
st.header("Business Insights")

st.write(
    """
    Both models estimate a negative relationship between price
    changes and sales movement changes.

    The filtered model produces a smaller negative coefficient
    than the baseline model.

    This indicates that extreme observations have a substantial
    influence on the estimated relationship.
    """
)

st.warning(
    "The baseline and filtered models use different samples. "
    "The difference cannot be attributed solely to extreme "
    "observations without further analysis. Causal interpretation "
    "depends on the Double Machine Learning identification assumptions."
)

st.caption(
    "EconoCausal | Dynamic Pricing via Double Machine Learning"
)
