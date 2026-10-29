
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
    "Dashboard prototype. Product and demand values below are "
    "sample data, not validated research results."
)

# Sample product data for testing the dashboard
products = pd.DataFrame({
    "Product": [
        "Wireless Headphones",
        "Running Shoes",
        "Smart Watch",
        "Travel Backpack",
        "Bluetooth Speaker"
    ],
    "Category": [
        "Electronics",
        "Fashion",
        "Electronics",
        "Accessories",
        "Electronics"
    ],
    "Price (INR)": [2499, 3299, 4199, 1599, 1899],
    "Demand": ["High", "Medium", "High", "Low", "Medium"]
})

st.header("Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Sample Products", len(products))

with col2:
    st.metric(
        "Average Sample Price",
        f"₹{products['Price (INR)'].mean():,.0f}"
    )

with col3:
    st.metric(
        "High-Demand Sample Products",
        int((products["Demand"] == "High").sum())
    )

st.header("Sample Product Catalog")
st.dataframe(products, use_container_width=True)

st.header("DML Model Status")

st.write("Treatment: Price")
st.write("Outcome: Demand (proposed dashboard example)")
st.warning(
    "A validated DML estimate and confidence interval "
    "are not connected yet. No causal result is reported."
)

st.caption(
    "EconoCausal | Research dashboard prototype"
)

