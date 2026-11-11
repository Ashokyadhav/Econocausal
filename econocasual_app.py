import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
        "Data Cleaning and EDA",
        "Dataset Configuration",
        "Double Machine Learning",
        "Uplift Modelling",
        "Budget Optimization",
        "Business Recommendations"
    ]
)

st.sidebar.divider()

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

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

        elif uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        st.sidebar.success("Dataset loaded successfully!")

        st.sidebar.caption(f"File: {uploaded_file.name}")
        st.sidebar.caption(f"Rows: {len(df):,}")
        st.sidebar.caption(f"Columns: {len(df.columns)}")

    except Exception as e:
        st.sidebar.error(f"Unable to load dataset: {e}")

# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def prepare_features(data, columns):
    """Convert selected columns into model-ready features."""

    X = data[columns].copy()

    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(X.median(numeric_only=True))

    return X.astype(float)


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

    col3.metric(
        "Supported Formats",
        "CSV and Excel"
    )

    st.divider()

    st.subheader("Project Objectives")

    st.write("""
    - Analyse different compatible datasets.
    - Identify treatment and outcome variables.
    - Estimate treatment effects using Double Machine Learning.
    - Identify customers or observations with positive predicted uplift.
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

    st.info(
        "Models are trained using the currently uploaded dataset. "
        "Results depend on the selected variables and assumptions."
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

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

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

            st.subheader("Numerical Feature Distribution")

            selected_column = st.selectbox(
                "Choose a numerical column",
                numeric_columns
            )

            st.bar_chart(
                df[selected_column].value_counts().sort_index()
                .head(30)
            )

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if categorical_columns:

            st.subheader("Categorical Feature Distribution")

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

        st.subheader("Select Analysis Variables")

        columns = df.columns.tolist()

        treatment_column = st.selectbox(
            "Treatment Column",
            columns
        )

        treatment_values = df[treatment_column].dropna().unique().tolist()

        if len(treatment_values) >= 2:

            control_value = st.selectbox(
                "Control Group Value",
                treatment_values,
                index=0
            )

            treatment_value = st.selectbox(
                "Treatment Group Value",
                [
                    value for value in treatment_values
                    if value != control_value
                ]
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
                    if col not in [
                        treatment_column,
                        outcome_column
                    ]
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

                st.error(
                    "No numerical outcome column is available."
                )

        else:

            st.error(
                "Treatment column must contain at least two values."
            )

# --------------------------------------------------
# CHECK CONFIGURATION
# --------------------------------------------------

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

    if not all(
        key in st.session_state
        for key in required
    ):
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

    if df[features].shape[1] == 0:
        st.error("No control variables selected.")
        return None

    data = df[
        df[treatment_column].isin(
            [control_value, treatment_value]
        )
    ].copy()

    data["__treatment__"] = (
        data[treatment_column] == treatment_value
    ).astype(int)

    data["__outcome__"] = pd.to_numeric(
        data[outcome_column],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=["__treatment__", "__outcome__"]
    )

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

    if X.shape[1] == 0:
        st.error("Selected features could not be converted into model inputs.")
        return None

    if len(X) < 50:
        st.error("At least 50 usable observations are required for this demo.")
        return None

    return data, X

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

                if len(np.unique(d)) != 2:
                    st.error("Treatment must be binary.")
                    st.stop()

                if len(data) < 100:
                    st.warning(
                        "Small datasets may produce unstable estimates."
                    )

                ml_l = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )

                ml_m = RandomForestRegressor(
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

                st.subheader("DML Results")

                st.dataframe(
                    result.style.format("{:.5f}"),
                    use_container_width=True
                )

                lower, upper = model.confint().iloc[0].tolist()

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

elif section == "Uplift Modelling":

    st.header("Uplift Modelling")

    st.write(
        "T-Learner compares predicted outcomes under treatment "
        "and control conditions."
    )

    if st.button("Run Uplift Analysis"):

        prepared = get_analysis_data()

        if prepared is not None:

            data, X = prepared

            y = data["__outcome__"]
            d = data["__treatment__"]

            if not y.isin([0, 1]).all():

                st.error(
                    "This T-Learner implementation requires a binary "
                    "outcome containing only 0 and 1."
                )

            elif (
                y[d == 0].nunique() < 2
                or y[d == 1].nunique() < 2
            ):

                st.error(
                    "Both treatment and control groups must contain "
                    "both outcome classes (0 and 1)."
                )

            else:

                try:

                    X_train, X_test, y_train, y_test, d_train, d_test = (
                        train_test_split(
                            X,
                            y,
                            d,
                            test_size=0.30,
                            random_state=42,
                            stratify=d
                        )
                    )

                    control_model = LogisticRegression(
                        max_iter=1000
                    )

                    treatment_model = LogisticRegression(
                        max_iter=1000
                    )

                    control_model.fit(
                        X_train[d_train == 0],
                        y_train[d_train == 0]
                    )

                    treatment_model.fit(
                        X_train[d_train == 1],
                        y_train[d_train == 1]
                    )

                    p_control = control_model.predict_proba(
                        X_test
                    )[:, 1]

                    p_treatment = treatment_model.predict_proba(
                        X_test
                    )[:, 1]

                    uplift = p_treatment - p_control

                    uplift_results = pd.DataFrame({
                        "Predicted Control Probability": p_control,
                        "Predicted Treatment Probability": p_treatment,
                        "Predicted Uplift": uplift
                    })

                    st.session_state["uplift_results"] = uplift_results

                    st.subheader("Uplift Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Average Predicted Uplift",
                        f"{np.mean(uplift):.5f}"
                    )

                    col2.metric(
                        "Positive Uplift",
                        int((uplift > 0).sum())
                    )

                    col3.metric(
                        "Negative Uplift",
                        int((uplift < 0).sum())
                    )

                    st.dataframe(
                        uplift_results.head(20),
                        use_container_width=True
                    )

                    st.subheader("Predicted Uplift Distribution")

                    st.bar_chart(
                        uplift_results["Predicted Uplift"]
                        .value_counts(bins=20)
                        .sort_index()
                    )

                    st.info(
                        "These are model predictions, not automatically "
                        "validated individual causal effects."
                    )

                except Exception as e:

                    st.error(f"Uplift modelling failed: {e}")

# --------------------------------------------------
# BUDGET OPTIMIZATION
# --------------------------------------------------

elif section == "Budget Optimization":

    st.header("Budget Optimization")

    st.write(
        "Allocate a limited budget to observations with "
        "the highest positive predicted uplift."
    )

    uplift_results = st.session_state.get(
        "uplift_results"
    )

    if uplift_results is None:

        st.warning(
            "Run Uplift Modelling first."
        )

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

            max_customers = int(
                budget // coupon_cost
            )

            positive_customers = (
                uplift_results[
                    uplift_results["Predicted Uplift"] > 0
                ]
                .sort_values(
                    "Predicted Uplift",
                    ascending=False
                )
            )

            selected = positive_customers.head(
                max_customers
            )

            customers_targeted = len(selected)

            expected_incremental_purchases = (
                selected["Predicted Uplift"].sum()
            )

            budget_used = customers_targeted * coupon_cost

            efficiency = (
                expected_incremental_purchases / budget_used
                if budget_used > 0 else 0
            )

            st.subheader("Budget Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customers Targeted",
                customers_targeted
            )

            col2.metric(
                "Budget Used",
                f"{budget_used:.2f}"
            )

            col3.metric(
                "Expected Incremental Purchases",
                f"{expected_incremental_purchases:.3f}"
            )

            st.metric(
                "Estimated Purchases per Budget Unit",
                f"{efficiency:.5f}"
            )

            st.dataframe(
                selected,
                use_container_width=True
            )

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

    for i, recommendation in enumerate(
        recommendations,
        1
    ):
        st.markdown(
            f"**{i}.** {recommendation}"
        )

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
