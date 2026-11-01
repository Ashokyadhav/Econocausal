import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("EconoCausal")
st.subheader("Dynamic Pricing via Double Machine Learning")

st.info(
    "Research dashboard: preliminary DML results. "
    "The estimate below must be validated before causal interpretation."
)

st.header("Double Machine Learning Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Estimated Price Effect", "-89.73")

with col2:
    st.metric("Standard Error", "2.41")

with col3:
    st.metric("P-value", "< 0.001")

st.subheader("Model Details")

results = pd.DataFrame({
    "Item": [
        "Dataset",
        "Treatment",
        "Outcome",
        "Model",
        "95% Confidence Interval"
    ],
    "Details": [
        "Cleaned WCER2",
        "d_PRICE (change in price)",
        "d_MOVE (change in sales volume)",
        "DoubleMLPLR",
        "[-94.45, -85.00]"
    ]
})

st.dataframe(results, use_container_width=True, hide_index=True)

st.warning(
    "These values are from the existing model summary. "
    "Confirm the model specification and data preparation before "
    "reporting the estimate as a causal effect. "
    "This dashboard does not rerun the model."
)

st.caption(
    "EconoCausal | Dynamic Pricing via Double Machine Learning"
)

st.caption(
    "EconoCausal | Research dashboard prototype"
)

