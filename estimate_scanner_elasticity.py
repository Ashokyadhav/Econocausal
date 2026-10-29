import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

# Load data
df = pd.read_csv("store_scanner_pricing.csv")

# Filter to main UPC (1313000002) and clean positive moves/prices
df_clean = df[(df["UPC"] == 1313000002) & (df["MOVE"] > 0) & (df["PRICE"] > 0) & (df["PRICE"] < 10)].copy()

# Outcome Y: log(MOVE), Treatment T: log(PRICE)
df_clean["log_move"] = np.log(df_clean["MOVE"])
df_clean["log_price"] = np.log(df_clean["PRICE"])

# Confounders X: Store fixed effects and Week time trends
df_clean["week_sin"] = np.sin(2 * np.pi * df_clean["WEEK"] / 52.0)
df_clean["week_cos"] = np.cos(2 * np.pi * df_clean["WEEK"] / 52.0)
df_clean["week_norm"] = df_clean["WEEK"] / df_clean["WEEK"].max()

store_dummies = pd.get_dummies(df_clean["STORE"], prefix="store", drop_first=True, dtype=float)
X = pd.concat([df_clean[["week_sin", "week_cos", "week_norm"]], store_dummies], axis=1).values
Y = df_clean["log_move"].values
T = df_clean["log_price"].values
n = len(Y)

# 1. Naive OLS (without controlling for confounding)
T_mat = np.column_stack([np.ones(n), T])
beta_naive = np.linalg.pinv(T_mat.T @ T_mat) @ T_mat.T @ Y
naive_elasticity = beta_naive[1]

# 2. Chernozhukov DML with 5-fold cross-fitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)
q_hat = np.zeros(n)  # E[Y | X]
m_hat = np.zeros(n)  # E[T | X]

for tr_idx, val_idx in kf.split(X):
    # Outcome nuisance model
    m_y = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.1, random_state=42)
    m_y.fit(X[tr_idx], Y[tr_idx])
    q_hat[val_idx] = m_y.predict(X[val_idx])

    # Price nuisance model
    m_t = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.1, random_state=42)
    m_t.fit(X[tr_idx], T[tr_idx])
    m_hat[val_idx] = m_t.predict(X[val_idx])

# Neyman-Orthogonal Residuals
tilde_Y = Y - q_hat
tilde_T = T - m_hat

# Stage 2: Robinson Orthogonal OLS with HC3 Standard Errors
Z = np.column_stack([np.ones(n), tilde_T])
inv_ZtZ = np.linalg.pinv(Z.T @ Z)
alpha = inv_ZtZ @ Z.T @ tilde_Y
residuals = tilde_Y - Z @ alpha

# HC3 Robust Sandwich Estimator
hat_diag = np.sum(Z * (Z @ inv_ZtZ), axis=1)
hc3_weights = (residuals / (1.0 - np.clip(hat_diag, 0.0, 0.95))) ** 2
omega = Z.T @ (hc3_weights[:, None] * Z)
var_cov = inv_ZtZ @ omega @ inv_ZtZ
dml_elasticity = alpha[1]
dml_se = np.sqrt(var_cov[1, 1])
dml_z = dml_elasticity / dml_se
dml_p = 2.0 * (1.0 - stats.norm.cdf(abs(dml_z)))
dml_ci = (dml_elasticity - 1.96 * dml_se, dml_elasticity + 1.96 * dml_se)

print("=== CAUSAL PRICE ELASTICITY RESULTS ===")
print(f"Total analyzed observations: {n}")
print(f"Naive OLS Elasticity: {naive_elasticity:.3f}")
print(f"DML Causal Elasticity: {dml_elasticity:.3f} (SE: {dml_se:.3f}, 95% CI: [{dml_ci[0]:.3f}, {dml_ci[1]:.3f}])")
print(f"Z-statistic: {dml_z:.2f}, P-value: {dml_p:.4e}")
