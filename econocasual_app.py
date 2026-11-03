import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- MODEL RESULTS ----------------
baseline = {
    "observations": 20000,
    "coefficient": -97.0281,
    "se": 10.2362,
    "lower": -117.0907,
    "upper": -76.9655
}

filtered = {
    "observations": 19572,
    "coefficient": -27.3148,
    "se": 0.982,
    "lower": -29.2395,
    "upper": -25.3901
}

# ---------------- SIDEBAR ----------------
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
st.sidebar.caption("Cluster-aware inference: PAIR_ID")

# ---------------- HEADER ----------------
st.title("📊 EconoCausal")

st.subheader(
    "Dynamic Pricing via Double Machine Learning"
)

st.write(
    "An analytical dashboard exploring the relationship "
    "between price changes and sales movement changes."
)

st.info(
    "This dashboard displays saved cluster-aware DML results. "
    "It does not train or rerun machine learning models."
)

# ---------------- OVERVIEW ----------------
if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Effect",
        f'{baseline["coefficient"]:.2f}'
    )

    col2.metric(
        "Filtered Effect",
        f'{filtered["coefficient"]:.2f}'
    )

    col3.metric(
        "Filtered Standard Error",
        f'{filtered["se"]:.3f}'
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

    observations = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Observations": [
            baseline["observations"],
            filtered["observations"]
        ]
    })

    st.bar_chart(
        observations.set_index("Model")
    )

# ---------------- MODEL COMPARISON ----------------
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

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Estimated Price Effect")

    effects = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Estimated Effect": [
            baseline["coefficient"],
            filtered["coefficient"]
        ]
    })

    st.bar_chart(
        effects.set_index("Model")
    )

    st.subheader("Standard Error Comparison")

    errors = pd.DataFrame({
        "Model": ["Baseline", "Filtered"],
        "Standard Error": [
            baseline["se"],
            filtered["se"]
        ]
    })

    st.bar_chart(
        errors.set_index("Model")
    )

    st.subheader("Confidence Interval Comparison")

    intervals = pd.DataFrame({
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
        intervals,
        use_container_width=True,
        hide_index=True
    )

# ---------------- RESEARCH INTERPRETATION ----------------
elif section == "Research Interpretation":

    st.header("Research Interpretation")

    st.subheader("Main Findings")

    st.write(
        "Both models estimate a negative relationship between "
        "price changes and sales movement changes."
    )

    st.write(
        f'The baseline coefficient is '
        f'{baseline["coefficient"]:.2f}, while the filtered '
        f'coefficient is {filtered["coefficient"]:.2f}.'
    )

    st.write(
        "The filtered estimate is smaller in magnitude. "
        "This indicates that the estimated relationship is "
        "sensitive to the filtering procedure."
    )

    st.warning(
        "The two estimates use different samples. Further "
        "robustness checks are needed before making strong "
        "causal claims."
    )

    st.subheader("Key Takeaway")

    st.success(
        "The analysis identifies a negative estimated relationship "
        "between price changes and sales movement changes."
    )

# ---------------- METHODOLOGY ----------------
elif section == "Methodology":

    st.header("Project Methodology")

    steps = [
        "Data collection and cleaning using WCER2.",
        "Creation of consecutive-week changes.",
        "Definition of treatment, outcome and control variables.",
        "Preparation of baseline and filtered samples.",
        "Double Machine Learning using DoubleMLPLR.",
        "Cluster-aware inference using PAIR_ID.",
        "Comparison and interpretation of model estimates."
    ]

    for i, step in enumerate(steps, start=1):
        st.markdown(f"**Step {i}.** {step}")

    st.divider()

    st.subheader("Filtering Details")

    st.write(
        "The filtered sensitivity analysis uses a restricted "
        "range of outcome changes (d_MOVE), as defined in the "
        "project's analysis notebook."
    )

    st.info(
        "Baseline observations: 20,000\n\n"
        "Filtered observations: 19,572"
    )

    st.subheader("Model Configuration")

    config = pd.DataFrame({
        "Setting": [
            "Treatment",
            "Outcome",
            "Controls",
            "Estimator",
            "Cluster variable",
            "Inference"
        ],
        "Configuration": [
            "d_PRICE",
            "d_MOVE",
            "STORE, UPC, WEEK",
            "DoubleMLPLR",
            "PAIR_ID",
            "Cluster-aware"
        ]
    })

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The dashboard displays saved model results. "
        "It does not perform model training."
    )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "EconoCausal | Dynamic Pricing via Double Machine Learning"
)

st.caption(
    "Research dashboard | Preliminary model results"
)
