import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EconoCausal - Multi Dataset",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EconoCausal")
st.subheader("Multi-Dataset Causal Analysis Platform")

st.write(
    "Upload compatible datasets, explore the data, "
    "configure treatment and outcome variables, and "
    "perform causal and uplift analysis."
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Project Navigation")

section = st.sidebar.radio(
    "Select Section",
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

st.sidebar.divider()
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel file",
    type=["csv", "xlsx"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.caption(f"File: {uploaded_file.name}")
        st.sidebar.caption(f"Rows: {len(df):,}")
        st.sidebar.caption(f"Columns: {len(df.columns)}")

    except Exception as e:
        st.sidebar.error(f"Unable to load dataset: {e}")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def prepare_features(data, columns):
    X = data[columns].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(float)


def get_analysis_data():

    if df is None:
        st.error("Please upload a dataset.")
        return None

    required = [
        "treatment_column",
        "control_value",
        "treatment_value",
        "outcome_column",
        "features"
    ]

    if not all(key in st.session_state for key in required):
        st.warning(
            "Please configure your dataset in Dataset Configuration first."
        )
        return None

    treatment_column = st.session_state["treatment_column"]
    outcome_column = st.session_state["outcome_column"]
    control_value = st.session_state["control_value"]
    treatment_value = st.session_state["treatment_value"]
    features = st.session_state["features"]

    if not features:
        st.error("Select at least one pre-treatment control variable.")
        return None

    data = df[
        df[treatment_column].isin([control_value, treatment_value])
    ].copy()

    data["__treatment__"] = (
        data[treatment_column] == treatment_value
    ).astype(int)

    data["__outcome__"] = pd.to_numeric(
        data[outcome_column],
        errors="coerce"
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["__treatment__", "__outcome__"])

    if data["__treatment__"].nunique() != 2:
        st.error("Both treatment and control groups are required.")
        return None

    if data["__treatment__"].value_counts().min() < 10:
        st.error("Each group needs at least 10 usable observations.")
        return None

    if data["__outcome__"].nunique() < 2:
        st.error("Outcome must contain at least two different values.")
        return None

    X = prepare_features(data, features)

    if X.shape[1] == 0 or len(X) < 50:
        st.error("Insufficient usable data or features.")
        return None

    return data, X


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if section == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        f"{len(df):,}" if df is not None else "Not uploaded"
    )

    col2.metric(
        "Dataset Features",
        len(df.columns) if df is not None else 0
    )

    col3.metric("Supported Formats", "CSV and Excel")

    st.divider()

    st.subheader("Project Objectives")

    st.write("""
    - Analyse different compatible datasets.
    - Identify treatment and outcome variables.
    - Estimate treatment effects using Double Machine Learning.
    - Identify observations with positive predicted uplift.
    - Explore budget allocation opportunities.
    """)

    st.subheader("Project Workflow")

    workflow = [
        "Upload dataset",
        "Data cleaning and exploratory analysis",
        "Configure treatment and outcome variables",
        "Run Double Machine Learning",
        "Perform uplift modelling",
        "Optimize budget",
        "Generate recommendations"
    ]

    for i, item in enumerate(workflow, 1):
        st.markdown(f"**{i}.** {item}")


# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

elif section == "Executive Summary":

    st.header("Executive Summary")
    st.write(
        "A single view of the uploaded dataset and the latest results "
        "from DML, uplift modelling and budget optimization."
    )

    # Dataset-level metrics
    total_customers = len(df) if df is not None else 0

    treatment_groups = "Not configured"
    if (
        df is not None
        and "treatment_column" in st.session_state
        and st.session_state["treatment_column"] in df.columns
    ):
        treatment_groups = int(
            df[st.session_state["treatment_column"]].nunique(dropna=True)
        )

    dml_summary = st.session_state.get("dml_summary")
    uplift_summary = st.session_state.get("uplift_summary")
    budget_summary = st.session_state.get("budget_summary")

    row1 = st.columns(2)
    row1[0].metric(
        "Total Customers / Records",
        f"{total_customers:,}" if df is not None else "Upload dataset"
    )
    row1[1].metric("Treatment Groups", treatment_groups)

    row2 = st.columns(3)
    row2[0].metric(
        "Estimated Treatment Effect (ATE)",
        f"{dml_summary['ate']:.5f}" if dml_summary else "Run DML"
    )
    row2[1].metric(
        "Positive Uplift",
        f"{uplift_summary['positive_count']:,}" if uplift_summary else "Run Uplift"
    )
    row2[2].metric(
        "Budget Efficiency",
        f"{budget_summary['efficiency']:.5f}" if budget_summary else "Run Budget Optimization"
    )

    st.divider()
    st.subheader("Latest Analysis Details")

    if dml_summary:
        st.markdown(
            f"- **DML p-value:** {dml_summary['p_value']:.5f}\n"
            f"- **95% confidence interval:** "
            f"{dml_summary['ci_lower']:.5f} to {dml_summary['ci_upper']:.5f}"
        )
        if dml_summary["p_value"] < 0.05:
            st.success("The latest DML estimate is statistically significant at the 5% level.")
        else:
            st.info("The latest DML estimate is not statistically significant at the 5% level.")
    else:
        st.caption("DML results will appear here after you run Double Machine Learning.")

    if uplift_summary:
        st.markdown(
            f"- **Average predicted uplift:** {uplift_summary['average_uplift']:.5f}\n"
            f"- **Negative uplift observations:** {uplift_summary['negative_count']:,}"
        )
    else:
        st.caption("Uplift results will appear here after you run Uplift Modelling.")

    if budget_summary:
        st.markdown(
            f"- **Customers targeted:** {budget_summary['customers_targeted']:,}\n"
            f"- **Budget used:** {budget_summary['budget_used']:.2f}\n"
            f"- **Expected incremental purchases (model estimate):** "
            f"{budget_summary['expected_incremental_purchases']:.3f}"
        )
    else:
        st.caption("Budget results will appear here after you run Budget Optimization.")

    st.warning(
        "These figures summarize the latest analyses in this app session. "
        "Predicted uplift and budget efficiency are model-based estimates, "
        "not verified individual causal effects or guaranteed business returns."
    )


# --------------------------------------------------
# DATA CLEANING AND EDA
# --------------------------------------------------

elif section == "Data Cleaning and EDA":

    st.header("Data Cleaning and Exploratory Data Analysis")

    if df is None:
        st.warning("Please upload a CSV or Excel dataset.")

    else:
        st.subheader("Dataset Preview")

        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])

        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Quality")

        quality = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Missing Values",
            int(df.isnull().sum().sum())
        )

        col2.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        st.subheader("Numerical Summary")
        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            selected_column = st.selectbox(
                "Choose a numerical column",
                numeric_columns
            )

            st.bar_chart(
                df[selected_column].value_counts()
                .sort_index()
                .head(30)
            )

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if categorical_columns:
            selected_category = st.selectbox(
                "Choose a categorical column",
                categorical_columns
            )

            st.bar_chart(
                df[selected_category].value_counts().head(20)
            )


# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

elif section == "Dataset Configuration":

    st.header("Dataset Configuration")

    if df is None:
        st.warning("Upload a dataset first.")

    else:
        columns = df.columns.tolist()

        treatment_column = st.selectbox(
            "Treatment Column",
            columns
        )

        treatment_values = (
            df[treatment_column].dropna().unique().tolist()
        )

        if len(treatment_values) >= 2:

            control_value = st.selectbox(
                "Control Group Value",
                treatment_values
            )

            treatment_options = [
                value for value in treatment_values
                if value != control_value
            ]

            treatment_value = st.selectbox(
                "Treatment Group Value",
                treatment_options
            )

            outcome_options = [
                col for col in columns
                if col != treatment_column
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if outcome_options:

                outcome_column = st.selectbox(
                    "Outcome Column",
                    outcome_options
                )

                feature_options = [
                    col for col in columns
                    if col not in [treatment_column, outcome_column]
                ]

                selected_features = st.multiselect(
                    "Pre-treatment Control Variables",
                    feature_options
                )

                st.warning(
                    "Select only variables measured before treatment. "
                    "Avoid IDs, leakage variables and post-treatment variables."
                )

                if st.button("Save Configuration"):

                    st.session_state["treatment_column"] = treatment_column
                    st.session_state["control_value"] = control_value
                    st.session_state["treatment_value"] = treatment_value
                    st.session_state["outcome_column"] = outcome_column
                    st.session_state["features"] = selected_features

                    st.success("Configuration saved successfully.")

                    st.write("Treatment:", treatment_column)
                    st.write("Outcome:", outcome_column)
                    st.write("Control variables:", selected_features)

            else:
                st.error("No numerical outcome column is available.")

        else:
            st.error("Treatment column must contain at least two values.")


# --------------------------------------------------
# DOUBLE MACHINE LEARNING
# --------------------------------------------------

elif section == "Double Machine Learning":

    st.header("Double Machine Learning")

    st.write(
        "Estimate the average treatment effect using "
        "Double Machine Learning with cross-fitting."
    )

    if st.button("Run DML Analysis"):

        prepared = get_analysis_data()

        if prepared is not None:

            data, X = prepared

            try:
                from doubleml import DoubleMLData, DoubleMLPLR

                y = data["__outcome__"].to_numpy(dtype=float)
                d = data["__treatment__"].to_numpy(dtype=float)

                ml_l = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )

                ml_m = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )

                dml_data = DoubleMLData.from_arrays(
                    x=X.to_numpy(),
                    y=y,
                    d=d
                )

                model = DoubleMLPLR(
                    dml_data,
                    ml_l=ml_l,
                    ml_m=ml_m,
                    n_folds=5
                )

                with st.spinner("Training DML model..."):
                    model.fit()

                result = pd.DataFrame({
                    "ATE": [model.coef[0]],
                    "Standard Error": [model.se[0]],
                    "P-value": [model.pval[0]]
                })

                lower, upper = model.confint().iloc[0].tolist()
                st.session_state["dml_summary"] = {
                    "ate": float(model.coef[0]),
                    "standard_error": float(model.se[0]),
                    "p_value": float(model.pval[0]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper)
                }

                st.subheader("DML Results")

                st.dataframe(
                    result.style.format("{:.5f}"),
                    use_container_width=True
                )

                st.write(
                    f"95% Confidence Interval: "
                    f"{lower:.5f} to {upper:.5f}"
                )

                if model.pval[0] < 0.05:
                    st.success(
                        "The estimated effect is statistically significant "
                        "at the 5% level."
                    )
                else:
                    st.info(
                        "The analysis did not find statistically significant "
                        "evidence of a treatment effect at the 5% level."
                    )

                st.warning(
                    "Causal interpretation depends on treatment assignment, "
                    "pre-treatment controls and model assumptions."
                )

            except Exception as e:
                st.error(f"DML analysis failed: {e}")


# --------------------------------------------------
# UPLIFT MODELLING
# --------------------------------------------------
def run_uplift(data, X):

    t = data["_t"].to_numpy()
    y = data["_y"].to_numpy(dtype=float)

    if np.bincount(t).min() < 20:
        raise ValueError(
            "Each treatment group needs at least 20 observations."
        )

    control = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    treated = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=5,
        random_state=43,
        n_jobs=-1
    )

    control.fit(X[t == 0], y[t == 0])
    treated.fit(X[t == 1], y[t == 1])

    y0 = control.predict(X)
    y1 = treated.predict(X)

    uplift = y1 - y0

    return pd.DataFrame({
        "Control Prediction": y0,
        "Treatment Prediction": y1,
        "Predicted Uplift": uplift
    })
# --------------------------------------------------
# BUDGET OPTIMIZATION
# --------------------------------------------------
if section == "Budget Optimization":
    st.header("Budget Optimization")
     st.write(
        "Allocate a limited budget to observations with "
        "the highest positive predicted uplift."
    )

    uplift_results = st.session_state.get("uplift_results")

    if uplift_results is None:

        st.warning("Run Uplift Modelling first.")

    else:

        budget = st.number_input(
            "Available Budget",
            min_value=1.0,
            value=1000.0
        )

        coupon_cost = st.number_input(
            "Cost per Treatment",
            min_value=0.01,
            value=1.0
        )

        if st.button("Optimize Budget"):

            max_customers = int(budget // coupon_cost)

            positive_customers = (
                uplift_results[
                    uplift_results["Predicted Uplift"] > 0
                ]
                .sort_values(
                    "Predicted Uplift",
                    ascending=False
                )
            )

            selected = positive_customers.head(max_customers)

            customers_targeted = len(selected)

            expected_incremental_purchases = (
                selected["Predicted Uplift"].sum()
            )

            budget_used = customers_targeted * coupon_cost

            efficiency = (
                expected_incremental_purchases / budget_used
                if budget_used > 0 else 0
            )

            st.session_state["budget_summary"] = {
                "customers_targeted": int(customers_targeted),
                "budget_used": float(budget_used),
                "expected_incremental_purchases": float(expected_incremental_purchases),
                "efficiency": float(efficiency)
            }

            st.subheader("Budget Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Customers Targeted", customers_targeted)
            col2.metric("Budget Used", f"{budget_used:.2f}")

            col3.metric(
                "Expected Incremental Purchases",
                f"{expected_incremental_purchases:.3f}"
            )

            st.metric(
                "Estimated Purchases per Budget Unit",
                f"{efficiency:.5f}"
            )

            st.dataframe(selected, use_container_width=True)

            st.warning(
                "Budget calculations are model-based estimates. "
                "They do not represent verified profit or savings."
            )


# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------

elif section == "Business Recommendations":

    st.header("Business Recommendations")

    st.subheader("General Recommendations")

    recommendations = [
        "Use suitable pre-treatment variables for causal analysis.",
        "Verify how treatment groups were assigned.",
        "Compare model predictions with real business outcomes.",
        "Validate uplift predictions before targeting customers.",
        "Use actual treatment costs for budget allocation."
    ]

    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {recommendation}")

    st.subheader("Project Limitations")

    limitations = [
        "Not every dataset is suitable for causal analysis.",
        "Treatment and outcome variables must be correctly defined.",
        "Post-treatment variables should not be used as ordinary controls.",
        "Model predictions depend on data quality and assumptions.",
        "Further validation is needed before business deployment."
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.success(
        "EconoCausal provides a configurable workflow for "
        "exploring treatment effects, uplift and budget allocation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EconoCausal | Multi-Dataset Causal Analysis Platform"
)

st.caption(
    "Research application | Results require appropriate "
    "statistical and business validation."
)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, roc_auc_score
import statsmodels.api as sm

st.set_page_config(
    page_title="EconoCausal",
    page_icon="📊",
    layout="wide"
)

# ---------------- SAVED COUPON RESEARCH RESULTS ----------------

DML_RESULTS = pd.DataFrame({
    "Comparison": [
        "High-tier - Amount",
        "High-tier - Purchase",
        "Low-tier - Amount",
        "Low-tier - Purchase"
    ],
    "ATE": [-0.2110, -0.0084, -0.0778, -0.0036],
    "P-value": [0.5000, 0.3374, 0.3187, 0.2279],
})

UPLIFT_RESULTS = pd.DataFrame({
    "Strategy": ["Low-tier", "High-tier T1", "High-tier T2"],
    "Average Predicted Uplift": [0.01018, -0.00574, 0.01162],
})

SEGMENTATION = pd.DataFrame({
    "Strategy": ["Low-tier"] * 3 + ["High-tier T1"] * 3 + ["High-tier T2"] * 3,
    "Segment": [
        "Negative Uplift",
        "Low/Moderate Uplift",
        "High Uplift"
    ] * 3,
    "Customers": [3, 1506, 527, 663, 101, 55, 842, 71, 362],
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
    ],
})


# ---------------- HELPERS ----------------

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
        data[treatment_col].isin([control_value, treatment_value])
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
            "Not enough observations in each treatment group for cross-fitting."
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
        ml_y, X, y, cv=cv, n_jobs=None
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
    p_value = float(fit.pvalues[1])

    return {
        "ate": ate,
        "se": se,
        "p_value": p_value,
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
            "Each treatment group needs at least 20 rows for uplift modelling."
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


# ---------------- ENTERPRISE UI DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ec-bg: #071521;
    --ec-panel: #0d2232;
    --ec-panel2: #102b3c;
    --ec-border: #1c3a4b;
    --ec-text: #e8f1f5;
    --ec-muted: #91a8b6;
    --ec-teal: #22c6b2;
    --ec-blue: #5c9dff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 78% -15%,
            rgba(24,116,133,.19),
            transparent 38%
        ),
        #071521;
    color: var(--ec-text);
}

[data-testid="stHeader"] {
    background: rgba(7,21,33,.78);
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b1d2b 0%,
        #091925 100%
    );
    border-right: 1px solid #1c3a4b;
    min-width: 275px;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Manrope', sans-serif;
    color: #f1f7fa;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 10px;
    margin: 3px 0;
    color: #b8cbd5;
    font-size: .92rem;
    transition: all .15s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #132f40;
    color: #fff;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #102a3a;
    border: 1px dashed #2d5968;
    border-radius: 12px;
}

.block-container {
    max-width: 1500px;
    padding-top: 2.1rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -.035em;
    color: #f1f7fa;
}

h1 {
    font-size: 2.15rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 1.12rem !important;
    font-weight: 700 !important;
}

p, li {
    color: #c3d2da;
    line-height: 1.65;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#102a3b,#0c2130);
    border: 1px solid #1d4051;
    border-radius: 16px;
    padding: 19px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.12);
    min-height: 122px;
}

[data-testid="stMetricLabel"] {
    color: #9fb7c4 !important;
    font-size: .84rem !important;
}

[data-testid="stMetricValue"] {
    color: #f2f8fa !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 750;
}

[data-testid="stMetricDelta"] {
    font-size: .78rem;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: #1d3b4b !important;
    border-radius: 16px !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1d3b4b;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"] {
    background: #0c2030;
    border: 1px solid #1b394a;
    border-radius: 14px;
    padding: 10px;
}

.stButton > button,
[data-testid="stDownloadButton"] button {
    background: linear-gradient(100deg,#0e9f91,#22c6b2);
    color: #041c24;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: .58rem 1.1rem;
    box-shadow: 0 5px 18px rgba(34,198,178,.14);
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: #51d9c8;
    color: #041c24;
    border: 0;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background: #102a3a;
    border-color: #29495a;
    border-radius: 9px;
}

hr {
    border-color: #1d3b4b !important;
}

.ec-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 20px;
}

.ec-logo {
    width: 43px;
    height: 43px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg,#22c6b2,#397bd7);
    color: #061923;
    font-size: 21px;
    font-weight: 900;
    box-shadow: 0 8px 24px rgba(34,198,178,.2);
}

.ec-brand-name {
    font-family: 'Manrope', sans-serif;
    color: #f2f8fa;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -.05em;
}

.ec-brand-sub {
    color: #7895a5;
    font-size: .65rem;
    letter-spacing: .13em;
    font-weight: 700;
    margin-top: 1px;
}

.ec-hero {
    position: relative;
    overflow: hidden;
    padding: 27px 30px;
    margin: 0 0 25px;
    border: 1px solid #1d4657;
    border-radius: 20px;
    background: linear-gradient(
        112deg,
        rgba(15,48,65,.98),
        rgba(10,34,50,.96) 58%,
        rgba(10,57,65,.94)
    );
    box-shadow: 0 18px 45px rgba(0,0,0,.16);
}

.ec-hero:after {
    content: '';
    position: absolute;
    width: 250px;
    height: 250px;
    right: 3%;
    top: -140px;
    border-radius: 50%;
    border: 1px solid rgba(64,210,192,.2);
    box-shadow:
        0 0 0 25px rgba(64,210,192,.025),
        0 0 0 55px rgba(64,210,192,.02);
}

.ec-kicker {
    color: #48d7c4;
    font-size: .7rem;
    letter-spacing: .16em;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.ec-hero-title {
    color: #f4f9fb;
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    line-height: 1.18;
    font-weight: 800;
    letter-spacing: -.045em;
    margin: 0 0 9px;
}

.ec-hero-copy {
    color: #a9c1cc;
    font-size: .95rem;
    max-width: 780px;
    margin: 0;
}

.ec-pill {
    display: inline-block;
    margin-top: 17px;
    padding: 6px 10px;
    border: 1px solid rgba(72,215,196,.25);
    border-radius: 30px;
    background: rgba(34,198,178,.09);
    color: #7be4d5;
    font-size: .72rem;
    font-weight: 700;
}

.ec-section-label {
    color: #48d7c4;
    font-size: .69rem;
    letter-spacing: .14em;
    font-weight: 800;
    text-transform: uppercase;
    margin: 8px 0 7px;
}

.ec-card {
    height: 100%;
    padding: 17px 19px;
    border: 1px solid #1d4051;
    border-radius: 15px;
    background: linear-gradient(145deg,#102a3b,#0c2130);
}

.ec-card-title {
    color: #91aab8;
    font-size: .78rem;
    font-weight: 600;
}

.ec-card-value {
    color: #f1f7fa;
    font-family: 'Manrope', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 7px;
    letter-spacing: -.04em;
}

.ec-card-note {
    color: #56d7c4;
    font-size: .72rem;
    margin-top: 5px;
}

.ec-footer {
    color: #6f8b9a;
    font-size: .72rem;
    border-top: 1px solid #1d3b4b;
    margin-top: 35px;
    padding-top: 14px;
}

@media (max-width:800px) {
    .block-container { padding: 1rem; }
    .ec-hero { padding: 21px; }
    .ec-hero-title { font-size: 1.55rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

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
        "Upload Coupon_Impact_Clean.csv (optional)",
        type=["csv"]
    )

    try:
        research_df = load_data(uploaded)
    except Exception as exc:
        research_df = None
        st.sidebar.error(f"Could not read file: {exc}")

    page = st.sidebar.radio(
        "Research sections",
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

    st.sidebar.caption("Upload your own compatible dataset")

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
        "Analytics sections",
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


# ---------------- PAGE HERO ----------------

st.markdown(
    f"""
    <div class="ec-hero">
        <div class="ec-kicker">ECONOCAUSAL / {mode.upper()}</div>
        <div class="ec-hero-title">{page}</div>
        <p class="ec-hero-copy">
            Understand what drives customer behaviour.
            Measure treatment effects, discover incremental uplift,
            and make smarter campaign investment decisions.
        </p>
        <span class="ec-pill">
            ● &nbsp;
            {"COUPON IMPACT RESEARCH" if mode == "Research Dashboard" else "FLEXIBLE DATA WORKSPACE"}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# ================= RESEARCH DASHBOARD =================

if mode == "Research Dashboard":

    if page == "Overview":

        st.header("Coupon Impact Research Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Dataset Records", "12,959")
        c2.metric("Dataset Features", "8")
        c3.metric("Research Modules", "3")

        st.markdown("### Project objectives")

        st.write("""
        - Analyse customer purchasing behaviour.
        - Estimate coupon treatment effects using Double Machine Learning.
        - Identify customers with positive predicted uplift.
        - Compare coupon strategies.
        - Explore marketing budget allocation.
        """)

        st.markdown("### Project workflow")

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
            "Upload the cleaned coupon CSV in the sidebar to explore its rows and columns."
        )

        st.caption(
            "Technology: Python, Pandas, NumPy, Scikit-learn, "
            "DoubleML research workflow and Streamlit."
        )

    elif page == "Data Cleaning and EDA":

        st.header("Data Cleaning and Exploratory Data Analysis")

        if research_df is None:

            st.warning(
                "Upload Coupon_Impact_Clean.csv in the sidebar to view the dataset."
            )

        else:

            a, b = st.columns(2)

            a.metric("Rows", f"{len(research_df):,}")
            b.metric("Columns", len(research_df.columns))

            st.subheader("Dataset preview")

            st.dataframe(
                research_df.head(20),
                use_container_width=True
            )

            st.subheader("Data quality")

            quality = pd.DataFrame({
                "Column": research_df.columns,
                "Missing values": research_df.isna().sum().values,
                "Data type": research_df.dtypes.astype(str).values
            })

            st.dataframe(
                quality,
                use_container_width=True,
                hide_index=True
            )

            a, b = st.columns(2)

            a.metric(
                "Total missing values",
                int(research_df.isna().sum().sum())
            )

            b.metric(
                "Duplicate rows",
                int(research_df.duplicated().sum())
            )

            if "group" in research_df.columns:

                st.subheader("Experimental group distribution")

                st.bar_chart(
                    research_df["group"].value_counts()
                )

                if "purchase" in research_df.columns:

                    st.subheader("Purchase rate by group")

                    rates = (
                        research_df.groupby("group")["purchase"]
                        .mean()
                        .mul(100)
                        .rename("Purchase rate (%)")
                    )

                    st.bar_chart(rates)

                    st.dataframe(
                        rates.reset_index(),
                        use_container_width=True,
                        hide_index=True
                    )

                if "amount" in research_df.columns:

                    st.subheader("Average purchase amount by group")

                    st.bar_chart(
                        research_df.groupby("group")["amount"]
                        .mean()
                        .rename("Average amount")
                    )

    elif page == "Double Machine Learning":

        st.header("Double Machine Learning — Saved Coupon Results")

        st.write(
            "Double Machine Learning estimates coupon treatment effects "
            "while accounting for observed customer characteristics."
        )
