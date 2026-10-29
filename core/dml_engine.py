"""
EconoCausal - Double Machine Learning (DML) Engine
Implements Chernozhukov et al. (2018) Neyman-orthogonal score estimation
with K-fold cross-fitting for multi-tiered pricing treatments ($0, $10, $20).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats


class DoubleMachineLearningEngine:
    """
    Chernozhukov Double Machine Learning Engine for Heterogeneous Treatment Effects.
    Eliminates regularization bias and confounding bias through:
    1. Cross-fitting (K=5 folds)
    2. Neyman-orthogonal score residualization:
       tilde(Y) = Y - E[Y | X]
       tilde(T_d) = T_d - E[T_d | X]
    3. Robinson transformation / R-Learner causal effect estimation:
       tilde(Y) = sum_d tau_d(X) * tilde(T_d) + error
    """

    def __init__(self, n_folds: int = 5, random_seed: int = 42, l2_reg: float = 1.0):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.l2_reg = l2_reg
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.treatments = [10, 20]  # Baseline is 0
        self.is_fitted = False

        # Results containers
        self.ate_results: Dict[int, Dict[str, float]] = {}
        self.cate_coefficients: Dict[int, np.ndarray] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.cov_matrix: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], outcome_col: str = "converted", treatment_col: str = "discount_offered") -> "DoubleMachineLearningEngine":
        """
        Fits the DML engine using K-fold cross-fitting.
        """
        self.feature_names = feature_cols
        X_raw = df[feature_cols].values
        Y = df[outcome_col].values.astype(float)
        T = df[treatment_col].values

        n_samples = len(Y)
        X_scaled = self.scaler.fit_transform(X_raw)

        # Multi-treatment indicator columns (for d=10, d=20)
        W_10 = (T == 10).astype(float)
        W_20 = (T == 20).astype(float)
        W = np.column_stack([W_10, W_20])

        # Stage 1: Cross-fitting Nuisance Models
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        q_hat = np.zeros(n_samples)        # E[Y | X]
        m_hat = np.zeros((n_samples, 2))   # E[W_10 | X], E[W_20 | X]

        for train_idx, val_idx in kf.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]
            W_tr, W_val = W[train_idx], W[val_idx]

            # Outcome Nuisance Model: q(X) = E[Y | X]
            outcome_model = GradientBoostingRegressor(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.08,
                random_state=self.random_seed
            )
            outcome_model.fit(X_tr, Y_tr)
            q_hat[val_idx] = outcome_model.predict(X_val)

            # Treatment Nuisance Models: m_d(X) = E[W_d | X] (Propensities)
            for d_idx in range(2):
                prop_model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=self.random_seed + d_idx
                )
                prop_model.fit(X_tr, W_tr[:, d_idx])
                # Clip estimated propensity away from 0 and 1 for strict positivity
                m_hat[val_idx, d_idx] = np.clip(prop_model.predict(X_val), 0.01, 0.99)

        # Compute Neyman-Orthogonal Residuals
        tilde_Y = Y - q_hat
        tilde_W = W - m_hat

        # Store nuisance diagnostics
        outcome_r2 = 1.0 - np.sum(tilde_Y ** 2) / np.sum((Y - np.mean(Y)) ** 2)
        corr_before = np.corrcoef(Y, W[:, 1])[0, 1]
        corr_after = np.corrcoef(tilde_Y, tilde_W[:, 1])[0, 1]

        self.diagnostics = {
            "n_samples": n_samples,
            "outcome_r2": float(outcome_r2),
            "unadjusted_corr_y_t20": float(corr_before),
            "orthogonal_corr_y_t20": float(corr_after),
            "mean_propensity_10": float(np.mean(m_hat[:, 0])),
            "mean_propensity_20": float(np.mean(m_hat[:, 1])),
            "mean_baseline_outcome": float(np.mean(q_hat))
        }

        # Stage 2A: Average Treatment Effect (ATE) Estimation via Robinson Orthogonal OLS
        # Model: tilde(Y) = alpha_10 * tilde(W_10) + alpha_20 * tilde(W_20) + eta
        # Using heteroskedasticity-consistent HC3 standard errors
        Z = tilde_W
        inv_ZtZ = np.linalg.pinv(Z.T @ Z)
        alpha = inv_ZtZ @ Z.T @ tilde_Y
        residuals_ate = tilde_Y - Z @ alpha

        # HC3 Sandwich Variance Estimator: (Z'Z)^-1 * (Z' diag(e_i / (1-h_i))^2 Z) * (Z'Z)^-1
        hat_diag = np.sum(Z * (Z @ inv_ZtZ), axis=1)
        hc3_weights = (residuals_ate / (1.0 - np.clip(hat_diag, 0.0, 0.95))) ** 2
        omega = Z.T @ (hc3_weights[:, None] * Z)
        var_cov_ate = inv_ZtZ @ omega @ inv_ZtZ
        std_errors = np.sqrt(np.diag(var_cov_ate))

        for idx, treat in enumerate(self.treatments):
            coef = float(alpha[idx])
            se = float(std_errors[idx])
            z_score = coef / se if se > 1e-9 else 0.0
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

            self.ate_results[treat] = {
                "ate": coef,
                "std_error": se,
                "z_score": z_score,
                "p_value": p_val,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper)
            }

        # Stage 2B: Heterogeneous Treatment Effect (CATE / ITE) Estimation
        # Parameterize: tau_d(X) = theta_{d,0} + theta_d^T X
        # Linear R-Learner formulation:
        # Features for each treatment: [tilde_W_d, tilde_W_d * X_scaled]
        X_interactions_10 = tilde_W[:, 0:1] * np.column_stack([np.ones((n_samples, 1)), X_scaled])
        X_interactions_20 = tilde_W[:, 1:2] * np.column_stack([np.ones((n_samples, 1)), X_scaled])
        Design_Matrix = np.column_stack([X_interactions_10, X_interactions_20])

        n_p = Design_Matrix.shape[1]
        reg_eye = np.eye(n_p) * self.l2_reg
        reg_eye[0, 0] = 0.0  # Don't regularize intercept for d=10
        reg_eye[1 + len(feature_cols), 1 + len(feature_cols)] = 0.0  # Don't regularize intercept for d=20

        theta = np.linalg.solve(Design_Matrix.T @ Design_Matrix + reg_eye, Design_Matrix.T @ tilde_Y)

        dim_per_treat = 1 + len(feature_cols)
        self.cate_coefficients[10] = theta[:dim_per_treat]
        self.cate_coefficients[20] = theta[dim_per_treat:]

        # Store nuisance models for full inference
        self.final_outcome_model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=self.random_seed)
        self.final_outcome_model.fit(X_scaled, Y)

        self.final_prop_model_10 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=self.random_seed)
        self.final_prop_model_10.fit(X_scaled, W_10)

        self.final_prop_model_20 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=self.random_seed + 1)
        self.final_prop_model_20.fit(X_scaled, W_20)

        self.is_fitted = True
        return self

    def predict_cate(self, df_or_X: Any) -> Dict[str, np.ndarray]:
        """
        Predicts Conditional Average Treatment Effect (CATE) tau_10(X) and tau_20(X)
        as well as baseline counterfactual conversion probability Y_0(X).
        """
        if not self.is_fitted:
            raise ValueError("DML engine has not been fitted yet.")

        if isinstance(df_or_X, pd.DataFrame):
            X_raw = df_or_X[self.feature_names].values
        else:
            X_raw = np.array(df_or_X)

        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        X_scaled = self.scaler.transform(X_raw)
        n_samples = X_scaled.shape[0]

        X_expanded = np.column_stack([np.ones((n_samples, 1)), X_scaled])

        tau_10 = X_expanded @ self.cate_coefficients[10]
        tau_20 = X_expanded @ self.cate_coefficients[20]

        # Predicted baseline outcome Y(0)
        # Using Robinson decomposition: E[Y | X] = Y0(X) + tau_10(X)*m_10(X) + tau_20(X)*m_20(X)
        q_pred = self.final_outcome_model.predict(X_scaled)
        m_10_pred = np.clip(self.final_prop_model_10.predict(X_scaled), 0.0, 1.0)
        m_20_pred = np.clip(self.final_prop_model_20.predict(X_scaled), 0.0, 1.0)

        y0_pred = q_pred - tau_10 * m_10_pred - tau_20 * m_20_pred
        y0_pred = np.clip(y0_pred, 0.01, 0.98)

        y10_pred = np.clip(y0_pred + tau_10, 0.01, 0.99)
        y20_pred = np.clip(y0_pred + tau_20, 0.01, 0.99)

        return {
            "tau_10": tau_10,
            "tau_20": tau_20,
            "y0_prob": y0_pred,
            "y10_prob": y10_pred,
            "y20_prob": y20_pred
        }

    def classify_uplift_quadrants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies each customer into one of the 4 Causal Uplift Quadrants:
        1. Persuadables: tau_20 >= 0.12 (High positive causal response)
        2. Sure Things: y0_prob >= 0.65 and tau_20 < 0.12 (Converts anyway)
        3. Lost Causes: y0_prob < 0.25 and tau_20 < 0.12 (Doomed regardless)
        4. Sleeping Dogs: tau_20 < 0.0 (Negative causal reaction to promo)
        """
        preds = self.predict_cate(df)
        tau_20 = preds["tau_20"]
        tau_10 = preds["tau_10"]
        y0_prob = preds["y0_prob"]

        quadrants = []
        for i in range(len(df)):
            t20 = tau_20[i]
            y0 = y0_prob[i]

            if t20 < -0.02:
                quad = "Sleeping Dogs"
            elif t20 >= 0.12:
                quad = "Persuadables"
            elif y0 >= 0.60:
                quad = "Sure Things"
            elif y0 < 0.25:
                quad = "Lost Causes"
            else:
                quad = "Persuadables" if t20 > 0.06 else "Lost Causes"

            quadrants.append(quad)

        res = df.copy()
        res["pred_tau_10"] = np.round(tau_10, 4)
        res["pred_tau_20"] = np.round(tau_20, 4)
        res["pred_y0_prob"] = np.round(y0_prob, 4)
        res["pred_y10_prob"] = np.round(preds["y10_prob"], 4)
        res["pred_y20_prob"] = np.round(preds["y20_prob"], 4)
        res["uplift_quadrant"] = quadrants
        return res

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns structured summary of causal model estimation."""
        return {
            "ate_results": self.ate_results,
            "diagnostics": self.diagnostics,
            "feature_coefficients": {
                "$10_discount": {
                    feat: float(self.cate_coefficients[10][i + 1])
                    for i, feat in enumerate(self.feature_names)
                },
                "$20_discount": {
                    feat: float(self.cate_coefficients[20][i + 1])
                    for i, feat in enumerate(self.feature_names)
                }
            }
        }
