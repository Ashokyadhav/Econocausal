import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="EconoCausal",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# SAVED COUPON IMPACT RESEARCH RESULTS
# These are the existing values from the supplied application.
# =========================================================
DML_RESULTS = pd.DataFrame({
    "Comparison": [
        "High-tier - Amount",
        "High-tier - Purchase",
        "Low-tier - Amount",
        "Low-tier - Purchase"
    ],
    "ATE": [-0.2110, -0.0084, -0.0778, -0.0036],
    "P-value": [0.5000, 0.3374, 0.3187, 0.2279]
})

UPLIFT_RESULTS = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Average Predicted Uplift": [0.01018, -0.00574, 0.01162]
})

SEGMENTATION = pd.DataFrame({
    "Strategy": ["Low-tier"] * 3 + ["High-tier T1"] * 3 + ["High-tier T2"] * 3,
    "Segment": [
        "Negative Uplift", "Low/Moderate Uplift", "High Uplift"
    ] * 3,
    "Customers": [3, 1506, 527, 663, 101, 55, 842, 71, 362]
})

BUDGET_RESULTS = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Customers Targeted": [1000, 156, 433],
    "Budget Used": [1000, 156, 433],
    "Expected Incremental Purchases": [20.515836, 2.886234, 17.787456],
    "Purchases per Budget Unit": [0.020516, 0.018501, 0.041080]
})

# =========================================================
# UI DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ec-bg: #071521;
    --ec-panel: #102a3b;
    --ec-border: #1d4051;
    --ec-text: #e8f1f5;
    --ec-muted: #91a8b6;
    --ec-teal: #22c6b2;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 78% -15%, rgba(24,116,133,.19), transparent 38%), #071521;
    color: var(--ec-text);
}
[data-testid="stHeader"] { background: rgba(7,21,33,.78); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1d2b, #091925);
    border-right: 1px solid #1c3a4b;
    min-width: 275px;
}
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #f1f7fa; letter-spacing: -.03em; }
p, li { color: #c3d2da; line-height: 1.65; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#102a3b,#0c2130);
    border: 1px solid #1d4051; border-radius: 15px;
    padding: 18px; min-height: 115px;
}
[data-testid="stMetricLabel"] { color: #9fb7c4 !important; }
[data-testid="stMetricValue"] { color: #f2f8fa !important; font-family: 'Manrope', sans-serif; }
[data-testid="stDataFrame"] { border: 1px solid #1d3b4b; border-radius: 12px; }
.stButton > button, [data-testid="stDownloadButton"] button {
    background: linear-gradient(100deg,#0e9f91,#22c6b2); color: #041c24;
    border: 0; border-radius: 10px; font-weight: 700; padding: .58rem 1.1rem;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover { background: #51d9c8; color: #041c24; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: #102a3a; border-color: #29495a; border-radius: 9px;
}
.ec-brand { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
.ec-logo {
    width:43px; height:43px; border-radius:13px; display:flex; align-items:center;
    justify-content:center; background:linear-gradient(145deg,#22c6b2,#397bd7);
    color:#061923; font-size:19px; font-weight:900;
}
.ec-brand-name { color:#f2f8fa; font-family:'Manrope',sans-serif; font-size:1.35rem; font-weight:800; }
.ec-brand-sub { color:#7895a5; font-size:.65rem; letter-spacing:.13em; font-weight:700; }
.ec-hero {
    padding:27px 30px; margin-bottom:25px; border:1px solid #1d4657;
    border-radius:20px; background:linear-gradient(112deg,#0f3041,#0a2232 58%,#0a3941);
}
.ec-kicker { color:#48d7c4; font-size:.7rem; letter-spacing:.16em; font-weight:800; }
.ec-hero-title { color:#f4f9fb; font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; margin:8px 0; }
.ec-hero-copy { color:#a9c1cc; max-width:800px; }
.ec-pill { display:inline-block; margin-top:10px; padding:6px 10px; border:1px solid #28665f; border-radius:30px; color:#7be4d5; font-size:.72rem; }
hr { border-color:#1d3b4b !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def load_data(file):
    if file is None:
        return None
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")


def make_features(data, columns):
    x = data[columns].copy()
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
        else:
            x[col] = x[col].astype("string").fillna("Missing")
    x = pd.get_dummies(x, drop_first=True, dtype=float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    return x.astype(float)


def prepare_analysis(df, treatment_col, outcome_col, control_value,
                     treatment_value, feature_cols):
    needed = list(dict.fromkeys(
        [treatment_col, outcome_col] + list(feature_cols)
    ))
    data = df[needed].copy()
    data = data[data[treatment_col].isin(
        [control_value, treatment_value]
    )].copy()

    data["_t"] = (data[treatment_col] == treatment_value).astype(int)
    data["_y"] = pd.to_numeric(data[outcome_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["_t", "_y"])

    if data["_t"].nunique() != 2:
        raise ValueError("Both treatment and control groups must be present.")
    if data["_t"].value_counts().min() < 10:
        raise ValueError("Each group needs at least 10 usable observations.")
    if data["_y"].nunique() < 2:
        raise ValueError("The outcome must contain at least two different values.")

    X = make_features(data, feature_cols)
    if len(data) < 50 or X.shape[1] == 0:
        raise ValueError("Need at least 50 usable rows and one valid control feature.")
    return data, X


def run_dml(data, X):
    """Cross-fitted partialling-out estimate; HC3 assumes independent rows."""
    t = data["_t"].to_numpy(dtype=int)
    y = data["_y"].to_numpy(dtype=float)
    folds = min(5, int(np.bincount(t).min()))
    if folds < 2:
        raise ValueError("Not enough observations in each group for cross-fitting.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    ml_y = RandomForestRegressor(
        n_estimators=120, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    ml_t = RandomForestClassifier(
        n_estimators=120, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    y_hat = cross_val_predict(ml_y, X, y, cv=cv, n_jobs=None)
    t_hat = cross_val_predict(
        ml_t, X, t, cv=cv, method="predict_proba"
    )[:, 1]
    t_hat = np.clip(t_hat, 0.02, 0.98)

    fit = sm.OLS(
        y - y_hat, sm.add_constant(t - t_hat)
    ).fit(cov_type="HC3")

    ate = float(fit.params[1])
    se = float(fit.bse[1])
    return {
        "ate": ate,
        "se": se,
        "p_value": float(fit.pvalues[1]),
        "ci_low": ate - 1.96 * se,
        "ci_high": ate + 1.96 * se,
        "n": len(data),
        "folds": folds
    }


def run_uplift(data, X):
    """T-Learner supports binary classification or continuous regression outcomes."""
    t = data["_t"].to_numpy(dtype=int)
    y = data["_y"].to_numpy(dtype=float)

    if np.bincount(t).min() < 20:
        raise ValueError("Each treatment group needs at least 20 usable observations.")

    binary = set(np.unique(y)).issubset({0.0, 1.0})
    if binary:
        model0 = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        model1 = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5,
            random_state=43, n_jobs=-1
        )
        model0.fit(X.loc[t == 0], y[t == 0])
        model1.fit(X.loc[t == 1], y[t == 1])
        pred0 = model0.predict_proba(X)[:, 1]
        pred1 = model1.predict_proba(X)[:, 1]
        label0, label1 = "Control probability", "Treatment probability"
    else:
        model0 = RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        model1 = RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5,
            random_state=43, n_jobs=-1
        )
        model0.fit(X.loc[t == 0], y[t == 0])
        model1.fit(X.loc[t == 1], y[t == 1])
        pred0 = model0.predict(X)
        pred1 = model1.predict(X)
        label0, label1 = "Control prediction", "Treatment prediction"

    return pd.DataFrame({
        label0: pred0,
        label1: pred1,
        "Predicted Uplift": pred1 - pred0
    })


def show_dataset_eda(data):
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(data):,}")
    b.metric("Columns", len(data.columns))
    c.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    quality = pd.DataFrame({
        "Column": data.columns,
        "Data type": data.dtypes.astype(str).values,
        "Missing values": data.isna().sum().values,
        "Unique values": data.nunique(dropna=True).values
    })
    st.subheader("Data quality")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.subheader("Numerical summary")
    st.dataframe(data.describe(include="all").T, use_container_width=True)


def render_analysis_pages(data, prefix):
    if data is None:
        st.warning("Upload a dataset in the sidebar to begin.")
        return

    # Keep settings separately for each dashboard mode.
    key = prefix
    columns = data.columns.tolist()

    if st.session_state.get(f"{key}_config") is None:
        st.session_state[f"{key}_config"] = None

    if page == "Overview":
        st.header("Business Analytics Overview")
        a, b, c = st.columns(3)
        a.metric("Dataset records", f"{len(data):,}")
        b.metric("Dataset features", len(columns))
        c.metric("Missing values", f"{int(data.isna().sum().sum()):,}")
        st.markdown("### Workflow")
        st.write("1. Explore and clean the dataset.")
        st.write("2. Define treatment, control, outcome and pre-treatment controls.")
        st.write("3. Run Double Machine Learning.")
        st.write("4. Run uplift modelling.")
        st.write("5. Explore budget allocation estimates.")
        st.info("Choose Dataset Configuration in the sidebar to define your analysis.")

    elif page == "Executive Summary":
        st.header("Executive Summary")
        config = st.session_state.get(f"{key}_config")
        dml = st.session_state.get(f"{key}_dml")
        uplift = st.session_state.get(f"{key}_uplift")
        budget = st.session_state.get(f"{key}_budget")
        a, b, c = st.columns(3)
        a.metric("Records", f"{len(data):,}")
        b.metric("Treatment groups", "Configured" if config else "Not configured")
        c.metric("DML ATE", f"{dml['ate']:.4f}" if dml else "Not run")
        if uplift is not None:
            st.metric("Positive predicted uplift rows",
                      f"{int((uplift['Predicted Uplift'] > 0).sum()):,}")
        if budget:
            st.write(f"Budget used: {budget['budget_used']:.2f}")
            st.write(f"Estimated incremental outcome: {budget['expected_incremental']:.3f}")
        st.caption("Model estimates require statistical and business validation.")

    elif page == "Data Cleaning and EDA":
        st.header("Data Cleaning and Exploratory Data Analysis")
        show_dataset_eda(data)

    elif page == "Dataset Configuration":
        st.header("Dataset Configuration")
        st.caption("Choose columns based on your research question, not just their data types.")
        treatment_col = st.selectbox(
            "Treatment column", columns, key=f"{key}_treatment_col"
        )
        values = data[treatment_col].dropna().unique().tolist()
        if len(values) < 2:
            st.error("The selected treatment column needs at least two distinct values.")
            return
        control_value = st.selectbox(
            "Control group value", values, key=f"{key}_control_value"
        )
        treatment_values = [v for v in values if v != control_value]
        treatment_value = st.selectbox(
            "Treatment group value", treatment_values, key=f"{key}_treated_value"
        )
        outcome_options = [
            col for col in columns
            if col != treatment_col and pd.api.types.is_numeric_dtype(data[col])
        ]
        if not outcome_options:
            st.error("No numeric outcome column is available.")
            return
        outcome_col = st.selectbox(
            "Outcome column", outcome_options, key=f"{key}_outcome_col"
        )
        feature_options = [
            col for col in columns if col not in [treatment_col, outcome_col]
        ]
        features = st.multiselect(
            "Pre-treatment control variables",
            feature_options,
            key=f"{key}_features"
        )
        st.info(
            "Use variables measured before treatment. IDs such as STORE or UPC "
            "may need categorical/fixed-effect handling; do not treat their numeric "
            "codes as meaningful quantities. Do not include post-treatment variables."
        )
        if st.button("Save Configuration", key=f"{key}_save_config"):
            if not features:
                st.error("Select at least one suitable pre-treatment control variable.")
            else:
                st.session_state[f"{key}_config"] = {
                    "treatment_col": treatment_col,
                    "control_value": control_value,
                    "treatment_value": treatment_value,
                    "outcome_col": outcome_col,
                    "features": features
                }
                st.session_state[f"{key}_dml"] = None
                st.session_state[f"{key}_uplift"] = None
                st.session_state[f"{key}_budget"] = None
                st.success("Configuration saved. You can now run DML or uplift analysis.")
        saved = st.session_state.get(f"{key}_config")
        if saved:
            st.subheader("Saved configuration")
            st.json(saved)

    elif page in ["Double Machine Learning", "Uplift Modelling"]:
        config = st.session_state.get(f"{key}_config")
        if not config:
            st.warning("Configure and save your dataset first.")
            return
        if st.button(
            "Run DML Analysis" if page == "Double Machine Learning" else "Run Uplift Analysis",
            key=f"{key}_run_{page}"
        ):
            try:
                analysis_data, X = prepare_analysis(
                    data,
                    config["treatment_col"],
                    config["outcome_col"],
                    config["control_value"],
                    config["treatment_value"],
                    config["features"]
                )
                with st.spinner("Running analysis..."):
                    if page == "Double Machine Learning":
                        result = run_dml(analysis_data, X)
                        st.session_state[f"{key}_dml"] = result
                    else:
                        result = run_uplift(analysis_data, X)
                        st.session_state[f"{key}_uplift"] = result
                        st.session_state[f"{key}_budget"] = None
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")

        if page == "Double Machine Learning":
            result = st.session_state.get(f"{key}_dml")
            if result:
                st.subheader("DML Results")
                a, b, c = st.columns(3)
                a.metric("Average Treatment Effect", f"{result['ate']:.5f}")
                b.metric("Standard error", f"{result['se']:.5f}")
                c.metric("P-value", f"{result['p_value']:.5f}")
                st.write(
                    f"95% confidence interval: {result['ci_low']:.5f} "
                    f"to {result['ci_high']:.5f}"
                )
                st.write(f"Usable observations: {result['n']:,} | Cross-fitting folds: {result['folds']}")
                if result["p_value"] < 0.05:
                    st.success("The estimate is statistically significant at the 5% level.")
                else:
                    st.info("The estimate is not statistically significant at the 5% level.")
                st.warning(
                    "This implementation uses HC3 standard errors that assume independent rows. "
                    "For panel data with repeated store-product observations, use an appropriate "
                    "panel design and clustered inference before making causal claims."
                )
        else:
            result = st.session_state.get(f"{key}_uplift")
            if result is not None:
                st.subheader("Uplift Results")
                st.metric("Average predicted uplift",
                          f"{result['Predicted Uplift'].mean():.5f}")
                st.metric("Rows with positive predicted uplift",
                          f"{int((result['Predicted Uplift'] > 0).sum()):,}")
                st.dataframe(
                    result.sort_values("Predicted Uplift", ascending=False).head(100),
                    use_container_width=True
                )
                st.download_button(
                    "Download uplift results",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="uplift_results.csv",
                    mime="text/csv"
                )
                st.caption(
                    "Uplift values are model-based predictions, not verified individual causal effects. "
                    "Validate on held-out data before using them for targeting."
                )

    elif page == "Budget Optimization":
        st.header("Budget Optimization")
        uplift = st.session_state.get(f"{key}_uplift")
        if uplift is None:
            st.warning("Run Uplift Modelling first.")
            return
        budget = st.number_input(
            "Available budget", min_value=1.0, value=1000.0,
            key=f"{key}_budget_amount"
        )
        cost = st.number_input(
            "Cost per treatment", min_value=0.01, value=1.0,
            key=f"{key}_cost"
        )
        if st.button("Optimize Budget", key=f"{key}_optimize"):
            max_rows = int(budget // cost)
            positive = uplift[uplift["Predicted Uplift"] > 0].sort_values(
                "Predicted Uplift", ascending=False
            )
            selected = positive.head(max_rows).copy()
            used = len(selected) * cost
            expected = float(selected["Predicted Uplift"].sum())
            efficiency = expected / used if used else 0.0
            st.session_state[f"{key}_budget"] = {
                "customers_targeted": len(selected),
                "budget_used": used,
                "expected_incremental": expected,
                "efficiency": efficiency
            }
            st.subheader("Budget results")
            a, b, c = st.columns(3)
            a.metric("Rows selected", f"{len(selected):,}")
            b.metric("Budget used", f"{used:.2f}")
            c.metric("Expected incremental outcome", f"{expected:.3f}")
            st.metric("Estimated uplift per budget unit", f"{efficiency:.5f}")
            st.dataframe(selected, use_container_width=True)
            st.warning(
                "This is a model-based allocation illustration. It is not a guarantee "
                "of profit, savings or realized incremental sales."
            )
            st.download_button(
                "Download selected rows",
                selected.to_csv(index=False).encode("utf-8"),
                file_name="budget_allocation.csv",
                mime="text/csv"
            )

    elif page == "Business Recommendations":
        st.header("Business Recommendations")
        st.write("1. Define treatment and outcome from a clear business question.")
        st.write("2. Use only suitable pre-treatment controls.")
        st.write("3. Check treatment/control group sizes and overlap.")
        st.write("4. Validate DML and uplift estimates on held-out data.")
        st.write("5. For repeated store-product data, account for panel structure and clustered errors.")
        st.warning("Causal conclusions depend on research design, data quality and model assumptions.")


# =========================================================
# SIDEBAR NAVIGATION AND DATA UPLOAD
# =========================================================
st.sidebar.markdown("""
<div class="ec-brand">
    <div class="ec-logo">EC</div>
    <div>
        <div class="ec-brand-name">EconoCausal</div>
        <div class="ec-brand-sub">CAUSAL INTELLIGENCE</div>
    </div>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Choose dashboard",
    ["Research Dashboard", "Business Analytics"]
)
st.sidebar.divider()

if mode == "Research Dashboard":
    st.sidebar.caption("Coupon Impact Dataset")
    uploaded = st.sidebar.file_uploader(
        "Upload Coupon Impact CSV (optional)",
        type=["csv"],
        key="coupon_upload"
    )
    try:
        research_df = load_data(uploaded)
    except Exception as exc:
        research_df = None
        st.sidebar.error(f"Could not read file: {exc}")

    pages = [
        "Overview", "Data Cleaning and EDA", "Double Machine Learning",
        "Uplift Modelling", "Budget Optimization", "Business Recommendations"
    ]
    page = st.sidebar.radio("Research sections", pages, key="research_page")

else:
    st.sidebar.caption("Upload another compatible dataset")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="business_upload"
    )
    try:
        business_df = load_data(uploaded)
    except Exception as exc:
        business_df = None
        st.sidebar.error(f"Could not read file: {exc}")

    pages = [
        "Overview", "Executive Summary", "Data Cleaning and EDA",
        "Dataset Configuration", "Double Machine Learning",
        "Uplift Modelling", "Budget Optimization", "Business Recommendations"
    ]
    page = st.sidebar.radio("Analytics sections", pages, key="business_page")

# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
    <div class="ec-hero">
        <div class="ec-kicker">ECONOCAUSAL / {mode.upper()}</div>
        <div class="ec-hero-title">{page}</div>
        <div class="ec-hero-copy">
            Understand what drives outcomes. Measure treatment effects,
            estimate incremental uplift and explore smarter budget allocation.
        </div>
        <span class="ec-pill">
            {"COUPON IMPACT RESEARCH" if mode == "Research Dashboard" else "FLEXIBLE DATA WORKSPACE"}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# RESEARCH DASHBOARD: SAVED COUPON RESULTS
# =========================================================
if mode == "Research Dashboard":
    if page == "Overview":
        st.header("Coupon Impact Research Overview")
        a, b, c = st.columns(3)
        a.metric("Dataset records", "12,959")
        b.metric("Dataset features", "8")
        c.metric("Research modules", "3")
        st.markdown("### Project objectives")
        st.write("- Analyse customer purchasing behaviour.")
        st.write("- Estimate coupon treatment effects using Double Machine Learning.")
        st.write("- Identify customers with positive predicted uplift.")
        st.write("- Compare coupon strategies and explore budget allocation.")
        st.markdown("### Research workflow")
        for i, item in enumerate([
            "Data cleaning and EDA", "Treatment/control analysis",
            "Double Machine Learning", "Uplift modelling",
            "Customer segmentation", "Budget optimization",
            "Business recommendations"
        ], 1):
            st.markdown(f"**{i}.** {item}")
        st.info(
            "The saved research results below are the Coupon Impact results from the supplied app. "
            "Upload the cleaned coupon CSV to inspect its records."
        )

    elif page == "Data Cleaning and EDA":
        if research_df is None:
            st.warning("Upload the Coupon Impact CSV in the sidebar to inspect the data.")
        else:
            show_dataset_eda(research_df)
            if "group" in research_df.columns:
                st.subheader("Experimental group distribution")
                st.bar_chart(research_df["group"].value_counts())
                if "purchase" in research_df.columns:
                    st.subheader("Purchase rate by group")
                    rates = research_df.groupby("group")["purchase"].mean().mul(100)
                    st.bar_chart(rates.rename("Purchase rate (%)"))
                if "amount" in research_df.columns:
                    st.subheader("Average amount by group")
                    st.bar_chart(research_df.groupby("group")["amount"].mean())

    elif page == "Double Machine Learning":
        st.header("Double Machine Learning — Saved Coupon Results")
        st.write(
            "These are the saved Coupon Impact estimates from the supplied application. "
            "They are displayed as recorded; this page does not retrain the original coupon models."
        )
        st.dataframe(DML_RESULTS, use_container_width=True, hide_index=True)
        st.caption("The listed p-values are all above 0.05; these saved estimates are not statistically significant at the 5% level.")

    elif page == "Uplift Modelling":
        st.header("Uplift Modelling — Saved Coupon Results")
        st.dataframe(UPLIFT_RESULTS, use_container_width=True, hide_index=True)
        st.subheader("Customer segmentation")
        st.dataframe(SEGMENTATION, use_container_width=True, hide_index=True)
        st.caption("These are saved results from the Coupon Impact analysis, not newly calculated predictions.")

    elif page == "Budget Optimization":
        st.header("Budget Optimization — Saved Coupon Results")
        st.dataframe(BUDGET_RESULTS, use_container_width=True, hide_index=True)
        st.caption("Expected incremental purchases and efficiency are model-based estimates, not guaranteed outcomes.")

    elif page == "Business Recommendations":
        st.header("Business Recommendations")
        st.write("- Compare coupon strategies using the saved treatment-effect and uplift results.")
        st.write("- Validate customer targeting before deploying a campaign.")
        st.write("- Consider the High-tier T2 strategy for further evaluation because it has the highest saved purchases-per-budget-unit value.")
        st.warning("The recommendation is based only on the saved summary values and should be validated against the original experiment and business costs.")

# =========================================================
# BUSINESS ANALYTICS: USER-UPLOADED DATASETS
# =========================================================
else:
    render_analysis_pages(business_df, "business")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("EconoCausal | Multi-Dataset Causal Analysis Platform")
st.caption("Research application | Results require appropriate statistical and business validation.")
