```python
"""
EconoCausal: Dynamic Pricing via Double Machine Learning (DML)

Objective:
Estimate price elasticity using Double Machine Learning
and compare it with naive Ordinary Least Squares (OLS).

Dataset:
store_scanner_pricing.csv

Columns:
STORE, UPC, WEEK, MOVE, PRICE

Outcome Y:
log(MOVE)

Treatment T:
log(PRICE)

Controls X:
Store fixed effects and seasonal/time trends.
"""

import warnings

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "store_scanner_pricing.csv"
TARGET_UPC = 1313000002

N_SPLITS = 5
RANDOM_STATE = 42

PRICE_MIN = 0
PRICE_MAX = 10

N_ESTIMATORS = 45
MAX_DEPTH = 3
LEARNING_RATE = 0.1


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print(" ECONOCAUSAL: DOUBLE MACHINE LEARNING PRICE ELASTICITY")
print("=" * 65)

try:
    df = pd.read_csv(DATA_PATH)

except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset '{DATA_PATH}' not found. "
        "Place the CSV file in your working directory."
    )

required_columns = ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\n[1] DATASET OVERVIEW")
print("-" * 65)

print(f"Original dataset shape: {df.shape}")
print(f"Missing values:\n{df[required_columns].isnull().sum()}")


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Convert required columns to numeric.
# Invalid values become NaN and are removed below.

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter the target UPC and valid sales/price observations.

df_clean = df.loc[
    (df["UPC"] == TARGET_UPC)
    & (df["MOVE"] > 0)
    & (df["PRICE"] > PRICE_MIN)
    & (df["PRICE"] < PRICE_MAX)
].dropna(
    subset=required_columns
).copy()

# Sort for easier inspection and reproducibility.
df_clean = df_clean.sort_values(
    ["STORE", "WEEK"]
).reset_index(drop=True)

if df_clean.empty:
    raise ValueError(
        "No valid observations remain after filtering. "
        "Check the UPC, MOVE, PRICE, and dataset values."
    )

n = len(df_clean)

print(f"\nTarget UPC: {TARGET_UPC}")
print(f"Valid observations: {n}")
print(f"Unique stores: {df_clean['STORE'].nunique()}")
print(f"Unique weeks: {df_clean['WEEK'].nunique()}")

if n < N_SPLITS:
    raise ValueError(
        f"At least {N_SPLITS} observations are required "
        "for the selected cross-fitting configuration."
    )

if df_clean["PRICE"].nunique() < 2:
    raise ValueError(
        "Price has no variation. Price elasticity cannot "
        "be estimated from this sample."
    )


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n[2] FEATURE ENGINEERING")
print("-" * 65)

# Log transformation of outcome and treatment.
df_clean["log_move"] = np.log(df_clean["MOVE"])
df_clean["log_price"] = np.log(df_clean["PRICE"])

# Time-related features.
week_min = df_clean["WEEK"].min()
week_max = df_clean["WEEK"].max()

df_clean["week_sin"] = np.sin(
    2 * np.pi * df_clean["WEEK"] / 52.0
)

df_clean["week_cos"] = np.cos(
    2 * np.pi * df_clean["WEEK"] / 52.0
)

# Handle the special case where all observations have the same week.
if week_max > week_min:
    df_clean["week_norm"] = (
        (df_clean["WEEK"] - week_min)
        / (week_max - week_min)
    )
else:
    df_clean["week_norm"] = 0.0

# Store fixed effects.
store_dummies = pd.get_dummies(
    df_clean["STORE"].astype(str),
    prefix="store",
    drop_first=True,
    dtype=float
)

# Control matrix.
X_df = pd.concat(
    [
        df_clean[
            ["week_sin", "week_cos", "week_norm"]
        ].astype(float),
        store_dummies
    ],
    axis=1
)

X = X_df.to_numpy(dtype=float)

# Outcome and treatment.
Y = df_clean["log_move"].to_numpy(dtype=float)
T = df_clean["log_price"].to_numpy(dtype=float)

if not (
    np.isfinite(X).all()
    and np.isfinite(Y).all()
    and np.isfinite(T).all()
):
    raise ValueError(
        "Non-finite values detected in the model inputs."
    )

print(f"Number of control features: {X.shape[1]}")
print(f"Outcome: log(MOVE)")
print(f"Treatment: log(PRICE)")


# ============================================================
# 5. NAIVE OLS BASELINE
# ============================================================

print("\n[3] NAIVE OLS BASELINE")
print("-" * 65)

# Regress log sales directly on log price.
# This does not control for store or time confounding.

T_mat = np.column_stack(
    [np.ones(n), T]
)

beta_naive = (
    np.linalg.pinv(T_mat.T @ T_mat)
    @ T_mat.T
    @ Y
)

naive_intercept = beta_naive[0]
naive_elasticity = beta_naive[1]

naive_predictions = T_mat @ beta_naive

naive_rmse = np.sqrt(
    mean_squared_error(Y, naive_predictions)
)

naive_r2 = r2_score(Y, naive_predictions)

print(f"Naive OLS intercept: {naive_intercept:.4f}")
print(f"Naive OLS elasticity: {naive_elasticity:.4f}")
print(f"Naive OLS RMSE: {naive_rmse:.4f}")
print(f"Naive OLS R-squared: {naive_r2:.4f}")


# ============================================================
# 6. DOUBLE MACHINE LEARNING: CROSS-FITTING
# ============================================================

print("\n[4] DOUBLE MACHINE LEARNING")
print("-" * 65)

print(f"Cross-fitting folds: {N_SPLITS}")
print("Training nuisance models...")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# q(X) = E[Y | X]
# m(X) = E[T | X]

q_hat = np.zeros(n)
m_hat = np.zeros(n)

for fold, (tr_idx, val_idx) in enumerate(
    kf.split(X), start=1
):
    print(f"Processing fold {fold}/{N_SPLITS}...")

    # Outcome nuisance model.
    model_y = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )

    model_y.fit(
        X[tr_idx],
        Y[tr_idx]
    )

    q_hat[val_idx] = model_y.predict(
        X[val_idx]
    )

    # Treatment nuisance model.
    model_t = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )

    model_t.fit(
        X[tr_idx],
        T[tr_idx]
    )

    m_hat[val_idx] = model_t.predict(
        X[val_idx]
    )

print("Cross-fitting completed.")


# ============================================================
# 7. NEYMAN-ORTHOGONAL RESIDUALS
# ============================================================

print("\n[5] RESIDUALIZATION")
print("-" * 65)

# Remove the components explained by the controls.

tilde_Y = Y - q_hat
tilde_T = T - m_hat

print(
    f"Residualized outcome standard deviation: "
    f"{np.std(tilde_Y, ddof=1):.6f}"
)

print(
    f"Residualized treatment standard deviation: "
    f"{np.std(tilde_T, ddof=1):.6f}"
)

# The residualized treatment must contain variation.
denominator = np.dot(tilde_T, tilde_T)

if denominator <= 1e-12:
    raise ValueError(
        "Residualized price has insufficient variation. "
        "The DML elasticity cannot be estimated reliably."
    )


# ============================================================
# 8. SECOND-STAGE ORTHOGONAL REGRESSION
# ============================================================

print("\n[6] SECOND-STAGE ESTIMATION")
print("-" * 65)

# Estimate the coefficient without an intercept.
#
# DML moment:
# sum(tilde_T * (tilde_Y - theta * tilde_T)) = 0

dml_elasticity = (
    np.dot(tilde_T, tilde_Y)
    / denominator
)

# Second-stage residuals.
residuals = (
    tilde_Y
    - dml_elasticity * tilde_T
)

print(
    f"DML causal elasticity estimate: "
    f"{dml_elasticity:.6f}"
)


# ============================================================
# 9. HC3-STYLE ROBUST STANDARD ERROR
# ============================================================

print("\n[7] ROBUST INFERENCE")
print("-" * 65)

# This calculation provides a heteroskedasticity-robust
# HC3-style standard error for the second-stage coefficient.
#
# It does NOT automatically account for dependence between
# observations from the same store or across time.

leverage = (
    (tilde_T ** 2)
    / denominator
)

# Numerical protection for the leverage adjustment.
adjustment = np.maximum(
    1.0 - leverage,
    1e-8
)

hc3_squared_residuals = (
    residuals / adjustment
) ** 2

# HC3 sandwich variance for a single regressor.
variance = (
    np.sum(
        (tilde_T ** 2)
        * hc3_squared_residuals
    )
    / (denominator ** 2)
)

if variance < 0 or not np.isfinite(variance):
    raise ValueError(
        "Invalid variance estimate. "
        "Check the data and model specification."
    )

dml_se = np.sqrt(variance)

if dml_se <= 0 or not np.isfinite(dml_se):
    raise ValueError(
        "The estimated standard error is invalid."
    )

# Large-sample normal approximation.
dml_z = dml_elasticity / dml_se

dml_p = 2.0 * stats.norm.sf(
    abs(dml_z)
)

critical_value = stats.norm.ppf(0.975)

dml_ci_lower = (
    dml_elasticity
    - critical_value * dml_se
)

dml_ci_upper = (
    dml_elasticity
    + critical_value * dml_se
)


# ============================================================
# 10. NUISANCE MODEL DIAGNOSTICS
# ============================================================

print("\n[8] NUISANCE MODEL DIAGNOSTICS")
print("-" * 65)

# Out-of-fold predictions are used for these diagnostics.
outcome_rmse = np.sqrt(
    mean_squared_error(Y, q_hat)
)

treatment_rmse = np.sqrt(
    mean_squared_error(T, m_hat)
)

outcome_r2 = r2_score(
    Y, q_hat
)

treatment_r2 = r2_score(
    T, m_hat
)

print(f"Outcome model RMSE: {outcome_rmse:.4f}")
print(f"Outcome model R-squared: {outcome_r2:.4f}")
print(f"Price model RMSE: {treatment_rmse:.4f}")
print(f"Price model R-squared: {treatment_r2:.4f}")


# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print(" CAUSAL PRICE ELASTICITY RESULTS")
print("=" * 65)

print(f"Analyzed observations: {n}")
print(f"Unique stores: {df_clean['STORE'].nunique()}")
print(f"Unique weeks: {df_clean['WEEK'].nunique()}")

print("\nNAIVE OLS")
print("-" * 65)
print(f"Price elasticity: {naive_elasticity:.4f}")

print("\nDOUBLE MACHINE LEARNING")
print("-" * 65)

print(f"DML elasticity: {dml_elasticity:.4f}")
print(f"Standard error: {dml_se:.4f}")

print(
    f"95% Confidence interval: "
    f"[{dml_ci_lower:.4f}, {dml_ci_upper:.4f}]"
)

print(f"Z-statistic: {dml_z:.4f}")
print(f"P-value: {dml_p:.6g}")

if dml_p < 0.05:
    print(
        "Statistical result: The coefficient is statistically "
        "different from zero at the 5% level under the "
        "calculation's inference assumptions."
    )
else:
    print(
        "Statistical result: The coefficient is not "
        "statistically different from zero at the 5% level "
        "under the calculation's inference assumptions."
    )


# ============================================================
# 12. INTERPRETATION
# ============================================================

print("\nINTERPRETATION")
print("-" * 65)

if dml_elasticity < 0:
    print(
        "The estimated price elasticity is negative. "
        "Higher prices are associated with lower sales, "
        "conditional on the model's controls."
    )
elif dml_elasticity > 0:
    print(
        "The estimated price elasticity is positive. "
        "Higher prices are associated with higher sales, "
        "conditional on the model's controls. Investigate "
        "possible confounding and model specification."
    )
else:
    print(
        "The estimated price elasticity is approximately zero."
    )

print(
    "\nA 1% increase in price corresponds approximately to "
    f"a {dml_elasticity:.4f}% change in sales under the "
    "constant-elasticity model."
)

print(
    "\nIMPORTANT: This estimate is causal only if the required "
    "identification assumptions hold. The controls in this "
    "script may not capture every source of price-sales "
    "confounding."
)

print(
    "The reported HC3-style standard error does not account "
    "for arbitrary store-level or temporal dependence."
)

print("=" * 65)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "Model": "Naive OLS",
            "Elasticity": naive_elasticity,
            "Standard_Error": np.nan,
            "CI_Lower": np.nan,
            "CI_Upper": np.nan,
            "P_Value": np.nan,
            "Observations": n
        },
        {
            "Model": "Double Machine Learning",
            "Elasticity": dml_elasticity,
            "Standard_Error": dml_se,
            "CI_Lower": dml_ci_lower,
            "CI_Upper": dml_ci_upper,
            "P_Value": dml_p,
            "Observations": n
        }
    ]
)

results.to_csv(
    "econocausal_elasticity_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "'econocausal_elasticity_results.csv'"
)
```
