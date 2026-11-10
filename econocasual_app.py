import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# ECONOCAUSAL | UI/UX REDESIGN
# Original Coupon Impact results and dashboard functionality
# are preserved. This update changes presentation and layout.
# ============================================================

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
    st.caption("COUPON INTELLIGENCE PLATFORM")
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
    st.caption("EconoCausal · Coupon Impact Research")

# -----------------------------
# STYLING
# -----------------------------
if theme == "Dark":
    bg = "#0B1020"
    panel = "#121A2B"
    panel2 = "#172238"
    text = "#E8EEF9"
    muted = "#9AA9C2"
    border = "#26344D"
    accent = "#8B7CFF"
    accent2 = "#35C6B0"
else:
    bg = "#F4F7FB"
    panel = "#FFFFFF"
    panel2 = "#EEF2F8"
    text = "#172033"
    muted = "#65728A"
    border = "#E0E6F0"
    accent = "#6558D3"
    accent2 = "#079B87"

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
    .hero {{ background: linear-gradient(115deg, {panel}, {panel2}); border:1px solid {border}; border-radius:22px; padding:28px 30px; margin: 0 0 22px 0; position:relative; overflow:hidden; }}
    .hero:after {{ content:''; position:absolute; width:220px; height:220px; right:-70px; top:-100px; border-radius:50%; background: radial-gradient(circle, {accent}55, transparent 68%); }}
    .eyebrow {{ color:{accent2}; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:10px; }}
    .hero-title {{ font-family:Manrope; color:{text}; font-size:32px; font-weight:800; line-height:1.15; margin:0 0 9px 0; }}
    .hero-copy {{ color:{muted}; font-size:14px; max-width:760px; margin:0; }}
    .section-kicker {{ color:{accent2}; font-size:11px; letter-spacing:.14em; font-weight:700; text-transform:uppercase; margin:24px 0 7px 0; }}
    .metric-card {{ background:{panel}; border:1px solid {border}; border-radius:17px; padding:17px 19px; min-height:112px; }}
    .metric-label {{ color:{muted}; font-size:12px; font-weight:600; margin-bottom:10px; }}
    .metric-value {{ color:{text}; font-family:Manrope; font-size:27px; font-weight:800; line-height:1.15; }}
    .metric-note {{ color:{muted}; font-size:11px; margin-top:7px; }}
    .info-card {{ background:{panel}; border:1px solid {border}; border-radius:16px; padding:18px 20px; height:100%; }}
    .info-title {{ color:{text}; font-weight:700; font-size:15px; margin-bottom:7px; }}
    .info-copy {{ color:{muted}; font-size:13px; line-height:1.55; }}
    .step {{ display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-bottom:1px solid {border}; }}
    .step:last-child {{ border-bottom:0; }}
    .step-num {{ background:{panel2}; border:1px solid {border}; color:{accent}; border-radius:10px; min-width:29px; height:29px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; }}
    .step-text {{ color:{text}; font-size:13px; padding-top:5px; }}
    .callout {{ background:{panel2}; border:1px solid {border}; border-left:3px solid {accent}; border-radius:12px; padding:13px 16px; color:{muted}; font-size:13px; }}
    div[data-testid="stDataFrame"] {{ border:1px solid {border}; border-radius:13px; overflow:hidden; }}
    div[data-testid="stMetric"] {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:14px 16px; }}
    div[data-testid="stMetricLabel"] {{ color:{muted}; }}
    div[data-testid="stMetricValue"] {{ color:{text}; font-family:Manrope; }}
    .stRadio label {{ font-size:13px; }}
    .stSelectbox, .stFileUploader {{ color:{text}; }}
    hr {{ border-color:{border}; }}
    .footer {{ color:{muted}; font-size:11px; text-align:center; padding:22px 0 5px 0; }}
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

# -----------------------------
# PAGE: OVERVIEW
# -----------------------------
if section == "Overview":
    hero(
        "Understand the impact.\nOptimize every offer.",
        "An integrated research dashboard for coupon effectiveness, causal impact, customer uplift and marketing budget decisions.",
        "COUPON IMPACT · RESEARCH WORKSPACE"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("DATASET RECORDS", "12,959", "Coupon Impact research dataset")
    with c2:
        metric_card("DATASET FEATURES", "8", "Original project overview")
    with c3:
        metric_card("CORE MODULES", "3", "DML · Uplift · Budget")

    section_title("Research overview", "Move from customer data to evidence-based coupon decisions.")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("<div class='info-card'><div class='info-title'>Project objectives</div><div class='info-copy'>Analyse customer purchasing behaviour, estimate coupon treatment effects, identify customers with positive predicted uplift, compare coupon strategies and improve marketing budget allocation.</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'><div class='info-title'>Technology stack</div><div class='info-copy'>Python · Pandas · NumPy · Matplotlib · Scikit-learn · DoubleML · Streamlit · GitHub</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='info-card'><div class='info-title'>Project workflow</div>", unsafe_allow_html=True)
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
            st.markdown(f"<div class='step'><div class='step-num'>{i:02d}</div><div class='step-text'>{step}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='callout'>This dashboard displays the saved Coupon Impact research results. It does not train the original research models. Upload the cleaned CSV from the sidebar to explore its dataset.</div>", unsafe_allow_html=True)

# -----------------------------
# PAGE: DATA CLEANING AND EDA
# -----------------------------
elif section == "Data Cleaning and EDA":
    hero("Data quality, at a glance.", "Explore the cleaned Coupon Impact dataset, inspect missing values and understand customer group patterns.", "DATA EXPLORATION")
    if df is None:
        st.info("Upload Coupon_Impact_Clean.csv using the sidebar to view dataset analysis.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("ROWS", f"{df.shape[0]:,}", "Loaded dataset")
        with c2: metric_card("COLUMNS", str(df.shape[1]), "Available features")
        with c3: metric_card("DUPLICATE ROWS", f"{int(df.duplicated().sum()):,}", "Exact duplicate records")

        section_title("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        section_title("Data quality report")
        quality = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values,
            "Data Type": df.dtypes.astype(str).values
        })
        q1, q2 = st.columns(2)
        with q1: metric_card("MISSING VALUES", f"{int(df.isnull().sum().sum()):,}", "Across all columns")
        with q2: metric_card("COMPLETE CELLS", f"{int(df.size - df.isnull().sum().sum()):,}", "Non-missing values")
        st.dataframe(quality, use_container_width=True, hide_index=True)

        if "group" in df.columns:
            section_title("Customer group analysis")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Experimental group distribution**")
                st.bar_chart(df["group"].value_counts())
            with g2:
                if "purchase" in df.columns:
                    st.markdown("**Purchase rate by group**")
                    purchase_rates = df.groupby("group")["purchase"].mean().mul(100).rename("Purchase Rate (%)")
                    st.bar_chart(purchase_rates)
                    st.dataframe(purchase_rates.reset_index(), use_container_width=True, hide_index=True)
            if "amount" in df.columns:
                st.markdown("**Average purchase amount by group**")
                average_amount = df.groupby("group")["amount"].mean().rename("Average Amount")
                st.bar_chart(average_amount)

# -----------------------------
# PAGE: DOUBLE MACHINE LEARNING
# -----------------------------
elif section == "Double Machine Learning":
    hero("Measure the treatment effect.", "Review the saved Average Treatment Effect estimates and their statistical significance across coupon comparisons.", "CAUSAL IMPACT · DOUBLE MACHINE LEARNING")
    st.markdown("Double Machine Learning estimates the average effect of coupon treatment on customer purchasing behaviour while accounting for relevant observed factors.")

    section_title("Model configuration")
    config = pd.DataFrame({
        "Parameter": ["Estimator", "Outcome Model", "Treatment Model", "Cross Validation", "Outcome Variables"],
        "Value": ["DoubleMLPLR", "Random Forest Regressor", "Logistic Regression", "5 Folds", "Purchase and Amount"]
    })
    st.dataframe(config, use_container_width=True, hide_index=True)

    section_title("Average treatment effect results")
    m1, m2, m3 = st.columns(3)
    with m1: metric_card("COMPARISONS", "4", "Saved DML estimates")
    with m2: metric_card("SIGNIFICANT RESULTS", "0", "At p < 0.05")
    with m3: metric_card("LOWEST P-VALUE", "0.2279", "Low-tier · Purchase")
    st.dataframe(dml_results.style.format({"ATE": "{:.4f}", "P-value": "{:.4f}"}), use_container_width=True, hide_index=True)

    section_title("ATE comparison")
    st.bar_chart(dml_results.set_index("Comparison")["ATE"])
    st.markdown("**Statistical interpretation**")
    st.write("All four comparisons have p-values greater than 0.05. Therefore, these analyses did not find statistically significant evidence of a coupon treatment effect. This does not prove that coupons have no effect.")
    st.warning("Causal interpretation depends on valid treatment assignment, appropriate pre-treatment controls and model assumptions.")

# -----------------------------
# PAGE: UPLIFT MODELLING
# -----------------------------
elif section == "Uplift Modelling":
    hero("Find customers worth targeting.", "Compare predicted uplift across coupon strategies and inspect the saved customer segmentation results.", "CUSTOMER INTELLIGENCE · UPLIFT MODELLING")
    st.markdown("Uplift modelling estimates how customer purchase probability may change when a customer receives a coupon.")

    section_title("How the T-Learner works")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='info-card'><div class='info-title'>01 · Control model</div><div class='info-copy'>Predicts purchase probability without treatment.</div></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='info-card'><div class='info-title'>02 · Treatment model</div><div class='info-copy'>Predicts purchase probability with treatment.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='callout'>Uplift Score = Treatment Prediction − Control Prediction</div>", unsafe_allow_html=True)

    section_title("Strategy comparison")
    st.dataframe(uplift_results.style.format({"Average Predicted Uplift": "{:.5f}"}), use_container_width=True, hide_index=True)
    st.bar_chart(uplift_results.set_index("Strategy")["Average Predicted Uplift"])

    section_title("Customer segmentation")
    st.dataframe(segmentation, use_container_width=True, hide_index=True)
    segment_chart = segmentation.pivot(index="Strategy", columns="Segment", values="Customers")
    st.bar_chart(segment_chart)
    st.info("Uplift scores are model predictions. Their causal interpretation requires validation.")

# -----------------------------
# PAGE: BUDGET OPTIMIZATION
# -----------------------------
elif section == "Budget Optimization":
    hero("Allocate budget with intent.", "Compare the saved coupon strategies by estimated incremental purchases and budget efficiency.", "MARKETING STRATEGY · BUDGET OPTIMIZATION")
    st.markdown("This module ranks customers using predicted uplift scores and compares coupon strategies under an illustrative budget scenario.")

    section_title("Budget allocation results")
    st.dataframe(budget_results.style.format({"Expected Incremental Purchases": "{:.2f}", "Purchases per Budget Unit": "{:.5f}"}), use_container_width=True, hide_index=True)

    best_strategy = budget_results.loc[budget_results["Purchases per Budget Unit"].idxmax()]
    highest_volume = budget_results.loc[budget_results["Expected Incremental Purchases"].idxmax()]
    x, y = st.columns(2)
    with x:
        metric_card("HIGHEST ESTIMATED EFFICIENCY", best_strategy["Strategy"], f"{best_strategy['Purchases per Budget Unit']:.5f} purchases per budget unit")
    with y:
        metric_card("HIGHEST ESTIMATED VOLUME", highest_volume["Strategy"], f"{highest_volume['Expected Incremental Purchases']:.2f} expected incremental purchases")

    section_title("Expected incremental purchases")
    st.bar_chart(budget_results.set_index("Strategy")["Expected Incremental Purchases"])
    section_title("Estimated budget efficiency")
    st.bar_chart(budget_results.set_index("Strategy")["Purchases per Budget Unit"])
    st.warning("Budget and coupon costs are illustrative assumptions. These results are not actual profit or savings estimates.")

# -----------------------------
# PAGE: BUSINESS RECOMMENDATIONS
# -----------------------------
elif section == "Business Recommendations":
    hero("Turn analysis into action.", "A concise view of the findings, practical recommendations and limitations of the current coupon analysis.", "DECISION SUPPORT")

    section_title("Key findings")
    findings = [
        "High-tier T2 has the highest average predicted uplift.",
        "High-tier T2 has the highest estimated purchases per budget unit in the illustrative scenario.",
        "Low-tier has the highest estimated incremental purchase volume in the budget scenario.",
        "DML results did not show statistically significant treatment effects in the four comparisons."
    ]
    for i, item in enumerate(findings, 1):
        st.markdown(f"<div class='step'><div class='step-num'>{i:02d}</div><div class='step-text'>{item}</div></div>", unsafe_allow_html=True)

    section_title("Recommendations")
    recommendations = [
        "Consider uplift-based customer targeting.",
        "Prioritize customers with positive predicted uplift.",
        "Compare coupon strategies using actual coupon costs.",
        "Validate model predictions through controlled experiments.",
        "Reassess budget allocation using real business outcomes."
    ]
    for item in recommendations:
        st.markdown(f"- {item}")

    section_title("Project limitations")
    limitations = [
        "Treatment assignment and experiment design need verification.",
        "Pre-treatment timing of predictor variables must be confirmed.",
        "Uplift predictions are not automatically causal effects.",
        "Budget values and coupon costs are illustrative.",
        "Further validation is required before business deployment."
    ]
    for item in limitations:
        st.markdown(f"- {item}")
    st.success("EconoCausal combines data analysis, causal machine learning and uplift modelling to support informed coupon allocation.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<div class='footer'>ECONOCAUSAL · DYNAMIC PRICING & COUPON EFFECTIVENESS ANALYSIS<br>Research dashboard · Saved model results and budget scenarios are subject to validation.</div>", unsafe_allow_html=True)
