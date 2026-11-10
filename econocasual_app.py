
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import statsmodels.api as sm

st.set_page_config(
    page_title="EconoCausal",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 1. SAVED COUPON RESEARCH RESULTS
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
    "Strategy": ["Low-tier"] * 3
                + ["High-tier T1"] * 3
                + ["High-tier T2"] * 3,
    "Segment": [
        "Negative Uplift",
        "Low/Moderate Uplift",
        "High Uplift"
    ] * 3,
    "Customers": [
        3, 1506, 527,
        663, 101, 55,
        842, 71, 362
    ]
})

BUDGET_RESULTS = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Customers Targeted": [1000, 156, 433],
    "Budget Used": [1000, 156, 433],
    "Expected Incremental Purchases": [
        20.515836, 2.886234, 17.787456
    ],
    "Purchases per Budget Unit": [
        0.020516, 0.018501, 0.041080
    ]
})


# =========================================================
# 2. HELPER FUNCTIONS
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


def prepare_analysis(
    df,
    treatment_col,
    outcome_col,
    control_value,
    treatment_value,
    feature_cols
):
    needed = [treatment_col, outcome_col] + list(feature_cols)
    data = df[needed].copy()

    data = data[
        data[treatment_col].isin(
            [control_value, treatment_value]
        )
    ].copy()

    data["_t"] = (
        data[treatment_col] == treatment_value
    ).astype(int)

    data["_y"] = pd.to_numeric(
        data[outcome_col],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["_t", "_y"])

    if data["_t"].nunique() != 2:
        raise ValueError(
            "Both treatment and control groups must be present."
        )

    if data["_t"].value_counts().min() < 10:
        raise ValueError(
            "Each group needs at least 10 usable rows."
        )

    if data["_y"].nunique() < 2:
        raise ValueError(
            "The outcome must contain at least two different values."
        )

    X = make_features(data, feature_cols)

    if len(data) < 50 or X.shape[1] == 0:
        raise ValueError(
            "Need at least 50 usable rows and one valid feature."
        )

    return data, X


def run_dml(data, X):
    """Cross-fitted partialling-out estimate with HC3 standard errors."""

    t = data["_t"].to_numpy()
    y = data["_y"].to_numpy(dtype=float)

    folds = min(5, int(np.bincount(t).min()))

    if folds < 2:
        raise ValueError(
            "Not enough observations in each group for cross-fitting."
        )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    ml_y = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    ml_t = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    y_hat = cross_val_predict(
        ml_y, X, y, cv=cv
    )

    t_hat = cross_val_predict(
        ml_t,
        X,
        t,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    t_hat = np.clip(t_hat, 0.02, 0.98)

    y_res = y - y_hat
    t_res = t - t_hat

    fit = sm.OLS(
        y_res,
        sm.add_constant(t_res)
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
    t = data["_t"].to_numpy()
    y = data["_y"].to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            "Uplift modelling requires a binary outcome coded as 0 and 1."
        )

    if np.bincount(t).min() < 20:
        raise ValueError(
            "Each treatment group needs at least 20 rows."
        )

    control = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    treated = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        random_state=43,
        n_jobs=-1
    )

    control.fit(X[t == 0], y[t == 0])
    treated.fit(X[t == 1], y[t == 1])

    p0 = control.predict_proba(X)[:, 1]
    p1 = treated.predict_proba(X)[:, 1]

    uplift = p1 - p0

    return pd.DataFrame({
        "Control probability": p0,
        "Treatment probability": p1,
        "Predicted uplift": uplift
    })


# =========================================================
# 3. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("EconoCausal")

mode = st.sidebar.radio(
    "Choose Dashboard",
    [
        "Research Dashboard",
        "Business Analytics"
    ]
)

st.sidebar.divider()

if mode == "Research Dashboard":

    st.sidebar.caption("Coupon Impact Dataset")

    uploaded = st.sidebar.file_uploader(
        "Upload Coupon_Impact_Clean.csv (optional)",
        type=["csv"]
    )

    try:
        research_df = load_data(uploaded)
    except Exception as exc:
        research_df = None
        st.sidebar.error(f"Could not read file: {exc}")

    page = st.sidebar.radio(
        "Research Sections",
        [
            "Overview",
            "Data Cleaning and EDA",
            "Double Machine Learning",
            "Uplift Modelling",
            "Budget Optimization",
            "Business Recommendations"
        ]
    )

else:

    st.sidebar.caption("Upload your own dataset")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    try:
        business_df = load_data(uploaded)
    except Exception as exc:
        business_df = None
        st.sidebar.error(f"Could not read file: {exc}")

    page = st.sidebar.radio(
        "Analytics Sections",
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


# =========================================================
# 4. MAIN HEADER
# =========================================================

st.title("📊 EconoCausal")

st.caption(
    "Dynamic Pricing, Coupon Effectiveness and Causal Analysis"
)


# =========================================================
# 5. RESEARCH DASHBOARD — COUPON PROJECT
# =========================================================

if mode == "Research Dashboard":

    if page == "Overview":

        st.header("Coupon Impact Research Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Dataset Records", "12,959")
        c2.metric("Dataset Features", "8")
        c3.metric("Research Modules", "3")

        st.markdown("### Project Objectives")

        st.write("""
        - Analyse customer purchasing behaviour.
        - Estimate coupon treatment effects using Double Machine Learning.
        - Identify customers with positive predicted uplift.
        - Compare coupon strategies.
        - Explore marketing budget allocation.
        """)

        st.markdown("### Project Workflow")

        workflow = [
            "Data cleaning and EDA",
            "Treatment and control group analysis",
            "Double Machine Learning",
            "Uplift modelling using T-Learner",
            "Customer segmentation",
            "Budget optimization",
            "Business recommendations"
        ]

        for i, item in enumerate(workflow, 1):
            st.markdown(f"**{i}.** {item}")

        st.info(
            "This page presents the saved Coupon Impact research results. "
            "Upload the cleaned coupon CSV to explore its rows and columns."
        )

        st.caption(
            "Technology: Python, Pandas, NumPy, Scikit-learn, "
            "DoubleML research workflow and Streamlit."
        )

    elif page == "Data Cleaning and EDA":

        st.header("Data Cleaning and Exploratory Data Analysis")

        if research_df is None:

            st.warning(
                "Upload Coupon_Impact_Clean.csv in the sidebar "
                "to view the dataset."
            )

        else:

            a, b = st.columns(2)

            a.metric("Rows", f"{len(research_df):,}")
            b.metric("Columns", len(research_df.columns))

            st.subheader("Dataset Preview")

            st.dataframe(
                research_df.head(20),
                use_container_width=True
            )

            st.subheader("Data Quality")

            quality = pd.DataFrame({
                "Column": research_df.columns,
                "Missing Values": research_df.isna().sum().values,
                "Data Type": research_df.dtypes.astype(str).values
            })

            st.dataframe(
                quality,
                use_container_width=True,
                hide_index=True
            )

            a, b = st.columns(2)

            a.metric(
                "Total Missing Values",
                int(research_df.isna().sum().sum())
            )

            b.metric(
                "Duplicate Rows",
                int(research_df.duplicated().sum())
            )

            if "group" in research_df.columns:

                st.subheader("Experimental Group Distribution")

                st.bar_chart(
                    research_df["group"].value_counts()
                )

                if "purchase" in research_df.columns:

                    st.subheader("Purchase Rate by Group")

                    rates = (
                        research_df.groupby("group")["purchase"]
                        .mean()
                        .mul(100)
                        .rename("Purchase Rate (%)")
                    )

                    st.bar_chart(rates)

                    st.dataframe(
                        rates.reset_index(),
                        use_container_width=True,
                        hide_index=True
                    )

                if "amount" in research_df.columns:

                    st.subheader("Average Purchase Amount by Group")

                    st.bar_chart(
                        research_df.groupby("group")["amount"]
                        .mean()
                        .rename("Average Amount")
                    )

    elif page == "Double Machine Learning":

        st.header("Double Machine Learning — Saved Coupon Results")

        st.write(
            "Double Machine Learning estimates coupon treatment effects "
            "while accounting for observed customer characteristics."
        )

        st.subheader("Model Configuration")

        config = pd.DataFrame({
            "Parameter": [
                "Estimator",
                "Outcome Model",
                "Treatment Model",
                "Cross Validation",
                "Outcomes"
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

        st.subheader("Average Treatment Effect Results")

        st.dataframe(
            DML_RESULTS.style.format({
                "ATE": "{:.4f}",
                "P-value": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            DML_RESULTS.set_index("Comparison")["ATE"]
        )

        st.markdown("### Statistical Interpretation")

        st.write("""
        All four comparisons have p-values greater than 0.05.

        These analyses therefore did not find statistically significant
        evidence of a coupon treatment effect.

        This does not prove that coupons have no effect.
        """)

        st.warning(
            "Causal interpretation depends on valid treatment assignment, "
            "appropriate pre-treatment controls and model assumptions."
        )

    elif page == "Uplift Modelling":

        st.header("Uplift Modelling — Saved Coupon Results")

        st.write("""
        The T-Learner estimates the difference between predicted
        purchase probability under treatment and control.
        """)

        st.markdown(
            "**Uplift = Treatment Prediction − Control Prediction**"
        )

        st.subheader("Strategy Comparison")

        st.dataframe(
            UPLIFT_RESULTS.style.format({
                "Average Predicted Uplift": "{:.5f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            UPLIFT_RESULTS.set_index("Strategy")
        )

        st.subheader("Customer Segmentation")

        st.dataframe(
            SEGMENTATION,
            use_container_width=True,
            hide_index=True
        )

        segment_chart = SEGMENTATION.pivot(
            index="Strategy",
            columns="Segment",
            values="Customers"
        )

        st.bar_chart(segment_chart)

        st.info(
            "Uplift values are saved model predictions. "
            "They are not guaranteed individual causal effects."
        )

    elif page == "Budget Optimization":

        st.header("Budget Optimization — Saved Coupon Results")

        st.dataframe(
            BUDGET_RESULTS.style.format({
                "Expected Incremental Purchases": "{:.2f}",
                "Purchases per Budget Unit": "{:.5f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Expected Incremental Purchases")

        st.bar_chart(
            BUDGET_RESULTS.set_index("Strategy")[
                "Expected Incremental Purchases"
            ]
        )

        st.subheader("Estimated Budget Efficiency")

        st.bar_chart(
            BUDGET_RESULTS.set_index("Strategy")[
                "Purchases per Budget Unit"
            ]
        )

        best = BUDGET_RESULTS.loc[
            BUDGET_RESULTS["Purchases per Budget Unit"].idxmax()
        ]

        volume = BUDGET_RESULTS.loc[
            BUDGET_RESULTS["Expected Incremental Purchases"].idxmax()
        ]

        a, b = st.columns(2)

        a.metric(
            "Highest Estimated Efficiency",
            best["Strategy"]
        )

        b.metric(
            "Highest Estimated Purchase Volume",
            volume["Strategy"]
        )

        st.warning(
            "Budget and coupon costs are illustrative assumptions, "
            "not verified profit or savings figures."
        )

    elif page == "Business Recommendations":

        st.header("Business Recommendations")

        st.markdown("### Key Findings")

        st.write("""
        1. High-tier T2 has the highest average predicted uplift.
        2. High-tier T2 has the highest estimated purchases per budget unit.
        3. Low-tier has the highest estimated incremental purchase volume.
        4. The four saved DML comparisons were not statistically significant
           at the 5% level.
        """)

        st.markdown("### Recommendations")

        st.write("""
        - Consider testing uplift-based customer targeting in a controlled campaign.
        - Validate model predictions with a fresh experiment before scaling coupon spend.
        - Track incremental purchases and actual coupon costs.
        - Avoid treating predicted uplift as a guaranteed outcome.
        """)


# =========================================================
# 6. BUSINESS ANALYTICS — OTHER DATASETS
# =========================================================

else:

    if page == "Overview":

        st.header("Business Analytics Overview")

        a, b, c = st.columns(3)

        a.metric(
            "Dataset Rows",
            f"{len(business_df):,}" if business_df is not None else "Not uploaded"
        )

        b.metric(
            "Dataset Columns",
            len(business_df.columns) if business_df is not None else 0
        )

        c.metric("Supported Formats", "CSV / Excel")

        st.write("""
        Upload a dataset, select treatment and outcome variables,
        and explore configurable DML, uplift and budget analysis.
        """)

        st.info(
            "For causal analysis, select pre-treatment control variables only. "
            "Do not include variables caused by the treatment."
        )

    elif page == "Executive Summary":

        st.header("Executive Summary")

        if business_df is None:

            st.info("Upload a CSV or Excel dataset in the sidebar to begin.")

        else:

            a, b = st.columns(2)

            a.metric("Records", f"{len(business_df):,}")
            b.metric("Features", len(business_df.columns))

            st.dataframe(
                business_df.head(10),
                use_container_width=True
            )

            for key, title in [
                ("dml_result", "Latest DML Result"),
                ("uplift_result", "Latest Uplift Result"),
                ("budget_result", "Latest Budget Result")
            ]:

                if key in st.session_state:
                    st.markdown(f"**{title}:** {st.session_state[key]}")

    elif page == "Data Cleaning and EDA":

        st.header("Data Cleaning and EDA")

        if business_df is None:

            st.warning("Upload a dataset in the sidebar first.")

        else:

            a, b = st.columns(2)

            a.metric("Rows", f"{len(business_df):,}")
            b.metric("Columns", len(business_df.columns))

            st.dataframe(
                business_df.head(20),
                use_container_width=True
            )

            quality = pd.DataFrame({
                "Column": business_df.columns,
                "Missing Values": business_df.isna().sum().values,
                "Data Type": business_df.dtypes.astype(str).values
            })

            st.dataframe(
                quality,
                use_container_width=True,
                hide_index=True
            )

            a, b = st.columns(2)

            a.metric(
                "Missing Values",
                int(business_df.isna().sum().sum())
            )

            b.metric(
                "Duplicate Rows",
                int(business_df.duplicated().sum())
            )

            numeric = business_df.select_dtypes(include=np.number)

            if not numeric.empty:

                st.subheader("Numeric Column Summary")

                st.dataframe(
                    numeric.describe().T,
                    use_container_width=True
                )

                chosen = st.selectbox(
                    "Choose a numeric column",
                    numeric.columns
                )

                st.bar_chart(
                    numeric[chosen].dropna()
                    .value_counts()
                    .sort_index()
                    .head(40)
                )

    elif page == "Dataset Configuration":

        st.header("Dataset Configuration")

        if business_df is None:

            st.warning("Upload a dataset in the sidebar first.")

        else:

            cols = list(business_df.columns)

            treatment_col = st.selectbox(
                "Treatment Column",
                cols,
                key="cfg_treatment"
            )

            vals = business_df[treatment_col].dropna().unique().tolist()

            if len(vals) < 2:

                st.error(
                    "Treatment column must contain at least two distinct values."
                )

            else:

                control_val = st.selectbox(
                    "Control Group Value",
                    vals,
                    key="cfg_control"
                )

                treatment_options = [
                    v for v in vals if v != control_val
                ]

                treatment_val = st.selectbox(
                    "Treatment Group Value",
                    treatment_options,
                    key="cfg_treated"
                )

                outcome_col = st.selectbox(
                    "Outcome Column",
                    [c for c in cols if c != treatment_col],
                    key="cfg_outcome"
                )

                feature_options = [
                    c for c in cols
                    if c not in [treatment_col, outcome_col]
                ]

                features = st.multiselect(
                    "Pre-treatment Control Variables",
                    feature_options,
                    default=feature_options[:min(5, len(feature_options))],
                    key="cfg_features"
                )

                if st.button("Save Configuration", type="primary"):

                    st.session_state["analysis_config"] = {
                        "treatment": treatment_col,
                        "control": control_val,
                        "treated": treatment_val,
                        "outcome": outcome_col,
                        "features": features
                    }

                    st.success(
                        "Configuration saved. Continue to DML or Uplift Modelling."
                    )

                st.caption(
                    "Use only variables measured before treatment. "
                    "For uplift modelling, the outcome must be binary 0/1."
                )

    elif page in [
        "Double Machine Learning",
        "Uplift Modelling",
        "Budget Optimization",
        "Business Recommendations"
    ]:

        st.header(page)

        if business_df is None:

            st.warning("Upload a dataset in the sidebar first.")

        elif "analysis_config" not in st.session_state:

            st.warning(
                "Configure your treatment, outcome and control variables "
                "in Dataset Configuration first."
            )

        else:

            cfg = st.session_state["analysis_config"]

            try:

                data, X = prepare_analysis(
                    business_df,
                    cfg["treatment"],
                    cfg["outcome"],
                    cfg["control"],
                    cfg["treated"],
                    cfg["features"]
                )

                st.caption(
                    f"Analysis rows: {len(data):,} | "
                    f"Control: {cfg['control']} | "
                    f"Treatment: {cfg['treated']} | "
                    f"Outcome: {cfg['outcome']}"
                )

                # ---------------- DML ----------------

                if page == "Double Machine Learning":

                    st.write(
                        "Cross-fitted partialling-out estimate using "
                        "Random Forest nuisance models and HC3 robust "
                        "standard errors."
                    )

                    if st.button(
                        "Run Double Machine Learning",
                        type="primary"
                    ):

                        result = run_dml(data, X)

                        st.session_state["dml_result"] = result

                    result = st.session_state.get("dml_result")

                    if result:

                        a, b, c = st.columns(3)

                        a.metric(
                            "Estimated ATE",
                            f"{result['ate']:.5f}"
                        )

                        b.metric(
                            "P-value",
                            f"{result['p_value']:.5f}"
                        )

                        c.metric(
                            "Observations",
                            f"{result['n']:,}"
                        )

                        st.write(
                            f"95% Confidence Interval: "
                            f"**{result['ci_low']:.5f} "
                            f"to {result['ci_high']:.5f}**"
                        )

                        if result["p_value"] < 0.05:
                            st.write(
                                "The estimate is statistically significant "
                                "at the 5% level."
                            )
                        else:
                            st.write(
                                "The estimate is not statistically significant "
                                "at the 5% level."
                            )

                        st.warning(
                            "This is an observational estimate. Its causal "
                            "interpretation depends on no important unobserved "
                            "confounding, valid pre-treatment controls, overlap "
                            "and correct implementation."
                        )

                # ---------------- UPLIFT ----------------

                elif page == "Uplift Modelling":

                    st.write(
                        "T-Learner: separate Random Forest classifiers "
                        "estimate purchase probability for control "
                        "and treatment groups."
                    )

                    if st.button(
                        "Run Uplift Modelling",
                        type="primary"
                    ):

                        uplift_df = run_uplift(data, X)

                        st.session_state["uplift_df"] = uplift_df

                        st.session_state["uplift_result"] = {
                            "positive_count": int(
                                (uplift_df["Predicted uplift"] > 0).sum()
                            ),
                            "mean_uplift": float(
                                uplift_df["Predicted uplift"].mean()
                            )
                        }

                    if "uplift_df" in st.session_state:

                        uplift_df = st.session_state["uplift_df"]

                        a, b = st.columns(2)

                        a.metric(
                            "Positive Predicted Uplift",
                            f"{(uplift_df['Predicted uplift'] > 0).sum():,}"
                        )

                        b.metric(
                            "Average Predicted Uplift",
                            f"{uplift_df['Predicted uplift'].mean():.5f}"
                        )

                        st.dataframe(
                            uplift_df.head(20),
                            use_container_width=True
                        )

                        st.bar_chart(
                            uplift_df["Predicted uplift"]
                            .clip(-1, 1)
                            .value_counts()
                            .sort_index()
                            .head(40)
                        )

                        st.info(
                            "Individual uplift estimates are model predictions, "
                            "not guaranteed individual causal effects."
                        )

                # ---------------- BUDGET ----------------

                elif page == "Budget Optimization":

                    if "uplift_df" not in st.session_state:

                        st.warning("Run Uplift Modelling first.")

                    else:

                        cost = st.number_input(
                            "Cost per Coupon / Treatment",
                            min_value=0.0,
                            value=1.0,
                            step=0.5
                        )

                        budget = st.number_input(
                            "Available Budget",
                            min_value=0.0,
                            value=1000.0,
                            step=100.0
                        )

                        if st.button(
                            "Optimize Budget",
                            type="primary"
                        ):

                            scores = (
                                st.session_state["uplift_df"]
                                .copy()
                                .sort_values(
                                    "Predicted uplift",
                                    ascending=False
                                )
                            )

                            scores = scores[
                                scores["Predicted uplift"] > 0
                            ]

                            max_people = (
                                int(budget // cost)
                                if cost > 0
                                else len(scores)
                            )

                            selected = scores.head(max_people)

                            spent = len(selected) * cost

                            expected = float(
                                selected["Predicted uplift"].sum()
                            )

                            st.session_state["budget_result"] = {
                                "targeted": len(selected),
                                "spent": spent,
                                "expected_incremental_outcomes": expected,
                                "efficiency": (
                                    expected / spent if spent else 0.0
                                )
                            }

                        if "budget_result" in st.session_state:

                            r = st.session_state["budget_result"]

                            a, b, c = st.columns(3)

                            a.metric(
                                "Customers Targeted",
                                f"{r['targeted']:,}"
                            )

                            b.metric(
                                "Budget Used",
                                f"{r['spent']:.2f}"
                            )

                            c.metric(
                                "Sum of Predicted Uplift",
                                f"{r['expected_incremental_outcomes']:.4f}"
                            )

                            st.caption(
                                "Predicted uplift per budget unit: "
                                f"{r['efficiency']:.5f}"
                            )

                            st.warning(
                                "This is a simplified allocation scenario. "
                                "It assumes equal treatment cost and uses "
                                "predicted uplift; it is not a profit forecast."
                            )

                # ---------------- RECOMMENDATIONS ----------------

                elif page == "Business Recommendations":

                    st.write("""
                    - Validate treatment effects with a randomized or carefully designed evaluation.
                    - Use pre-treatment variables only in causal models.
                    - Target positive predicted uplift only after out-of-sample validation.
                    - Compare incremental outcomes with actual campaign costs.
                    - Monitor performance across customer segments.
                    """)

            except Exception as exc:

                st.error(
                    f"Analysis could not be completed: {exc}"
                )


# =========================================================
# 7. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "EconoCausal | Coupon Research + Business Analytics"
)
