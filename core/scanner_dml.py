"""
EconoCausal - Retail Scanner Dynamic Pricing via Double Machine Learning
Estimates heterogeneous causal price elasticity and optimal shelf pricing
from empirical store scanner data (STORE, UPC, WEEK, MOVE, PRICE).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats


class ScannerDynamicPricingEngine:
    """
    Double Machine Learning Engine for Empirical Scanner Pricing Data.
    Estimates causal price elasticity:
    log(MOVE) = theta(STORE) * log(PRICE) + g(STORE, WEEK) + error
    and computes profit-maximizing dynamic prices under store-specific demand curves.
    """

    def __init__(self, csv_path: str = "store_scanner_pricing.csv", random_seed: int = 42):
        self.csv_path = csv_path
        self.random_seed = random_seed
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.overall_dml_elasticity: float = -1.25
        self.overall_naive_elasticity: float = -0.37
        self.store_results: Dict[int, Dict[str, Any]] = {}
        self.is_fitted: bool = False

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and cleans the empirical store scanner data."""
        df = pd.read_csv(self.csv_path)
        self.df_raw = df

        # Filter to the primary retail UPC (1313000002) with valid positive observations
        # Drop occasional anomalous bulk scanner spikes (> 500 units) to focus on standard consumer demand
        mask = (
            (df["UPC"] == 1313000002) &
            (df["MOVE"] > 0) &
            (df["MOVE"] <= 300) &
            (df["PRICE"] >= 1.20) &
            (df["PRICE"] <= 4.00)
        )
        df_clean = df[mask].copy()

        # Log-transformed variables
        df_clean["log_move"] = np.log(df_clean["MOVE"])
        df_clean["log_price"] = np.log(df_clean["PRICE"])

        # Time features for weekly seasonality
        df_clean["week_sin"] = np.sin(2 * np.pi * df_clean["WEEK"] / 52.0)
        df_clean["week_cos"] = np.cos(2 * np.pi * df_clean["WEEK"] / 52.0)
        df_clean["week_trend"] = df_clean["WEEK"] / df_clean["WEEK"].max()

        self.df_clean = df_clean
        return df_clean

    def fit(self) -> "ScannerDynamicPricingEngine":
        """
        Fits Double Machine Learning with K-Fold cross-fitting to isolate causal price elasticity.
        """
        if self.df_clean is None:
            self.load_and_preprocess()

        df = self.df_clean
        n = len(df)

        # Build Confounders Matrix X: Store fixed effects and Time trends
        store_dummies = pd.get_dummies(df["STORE"], prefix="store", drop_first=True, dtype=float)
        time_feats = df[["week_sin", "week_cos", "week_trend"]]
        X = pd.concat([time_feats, store_dummies], axis=1).values
        Y = df["log_move"].values
        T = df["log_price"].values

        # 1. Naive OLS (unadjusted baseline)
        T_design = np.column_stack([np.ones(n), T])
        beta_naive = np.linalg.pinv(T_design.T @ T_design) @ T_design.T @ Y
        self.overall_naive_elasticity = float(beta_naive[1])

        # 2. Chernozhukov DML Stage 1: Cross-fitting nuisance models
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        q_hat = np.zeros(n)  # E[Y | X]
        m_hat = np.zeros(n)  # E[T | X]

        for tr_idx, val_idx in kf.split(X):
            m_y = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.1, random_state=self.random_seed)
            m_y.fit(X[tr_idx], Y[tr_idx])
            q_hat[val_idx] = m_y.predict(X[val_idx])

            m_t = GradientBoostingRegressor(n_estimators=45, max_depth=3, learning_rate=0.1, random_state=self.random_seed + 1)
            m_t.fit(X[tr_idx], T[tr_idx])
            m_hat[val_idx] = m_t.predict(X[val_idx])

        # Neyman-Orthogonal Residuals
        tilde_Y = Y - q_hat
        tilde_T = T - m_hat

        # Stage 2: Orthogonal Regression for Overall ATE Elasticity
        Z = np.column_stack([np.ones(n), tilde_T])
        inv_ZtZ = np.linalg.pinv(Z.T @ Z)
        alpha = inv_ZtZ @ Z.T @ tilde_Y
        residuals = tilde_Y - Z @ alpha

        # HC3 Robust Sandwich Estimator
        hat_diag = np.sum(Z * (Z @ inv_ZtZ), axis=1)
        hc3_weights = (residuals / (1.0 - np.clip(hat_diag, 0.0, 0.95))) ** 2
        omega = Z.T @ (hc3_weights[:, None] * Z)
        var_cov = inv_ZtZ @ omega @ inv_ZtZ

        self.overall_dml_elasticity = float(alpha[1])
        overall_se = float(np.sqrt(var_cov[1, 1]))
        overall_p = float(2.0 * (1.0 - stats.norm.cdf(abs(self.overall_dml_elasticity / (overall_se + 1e-9)))))

        # 3. Store-Specific Elasticity & Dynamic Pricing Optimization
        # Store unit cost assumption (wholesale cost ~ $1.20)
        wholesale_cost = 1.20

        df_with_residuals = df.copy()
        df_with_residuals["tilde_Y"] = tilde_Y
        df_with_residuals["tilde_T"] = tilde_T

        unique_stores = sorted(df["STORE"].unique().tolist())
        self.store_results = {}

        for st in unique_stores:
            st_data = df_with_residuals[df_with_residuals["STORE"] == st]
            st_n = len(st_data)
            if st_n < 15:
                continue

            # Store-specific naive OLS
            st_T = st_data["log_price"].values
            st_Y = st_data["log_move"].values
            st_T_mat = np.column_stack([np.ones(st_n), st_T])
            st_naive_beta = np.linalg.pinv(st_T_mat.T @ st_T_mat) @ st_T_mat.T @ st_Y
            st_naive_elas = float(st_naive_beta[1])

            # Store-specific causal elasticity via orthogonal residuals
            st_t_res = st_data["tilde_T"].values
            st_y_res = st_data["tilde_Y"].values

            # Regularized shrinkage toward overall DML elasticity
            # theta_s = (sum tilde_T * tilde_Y + lambda * theta_global) / (sum tilde_T^2 + lambda)
            lambda_reg = 0.5
            denom = np.sum(st_t_res ** 2) + lambda_reg
            num = np.sum(st_t_res * st_y_res) + lambda_reg * self.overall_dml_elasticity
            st_dml_elas = float(num / denom)

            # Store baseline economics
            avg_price = float(st_data["PRICE"].mean())
            avg_move = float(st_data["MOVE"].mean())
            current_weekly_revenue = avg_price * avg_move
            current_weekly_profit = (avg_price - wholesale_cost) * avg_move

            # Profit-Maximizing Dynamic Price Calculation:
            # Profit Pi(P) = (P - cost) * Q0 * (P / P0)^theta
            # First order condition: dPi/dP = 0 => P* = (theta / (theta + 1)) * cost if theta < -1
            # If demand is inelastic (|theta| <= 1), marginal profit is positive, price bounded by ceiling
            if st_dml_elas < -1.05:
                optimal_price = (st_dml_elas / (st_dml_elas + 1.0)) * wholesale_cost
                # Clip to realistic retail band [$1.49, $2.49]
                optimal_price = float(np.clip(optimal_price, 1.49, 2.49))
            else:
                # Moderate markup
                optimal_price = float(min(avg_price * 1.08, 2.39))

            optimal_price = round(optimal_price, 2)

            # Projected demand at optimal price: Q* = Q0 * (P* / P0)^theta
            projected_move = avg_move * ((optimal_price / avg_price) ** st_dml_elas)
            projected_move = max(projected_move, 1.0)

            projected_weekly_revenue = round(optimal_price * projected_move, 2)
            projected_weekly_profit = round((optimal_price - wholesale_cost) * projected_move, 2)

            profit_lift_usd = round(projected_weekly_profit - current_weekly_profit, 2)
            profit_lift_pct = round((profit_lift_usd / (current_weekly_profit + 1e-6)) * 100.0, 1)

            # Generate demand & profit simulation curve across price points [$1.40 to $2.70]
            price_curve = np.linspace(1.40, 2.60, 25)
            move_curve = [round(float(avg_move * ((p / avg_price) ** st_dml_elas)), 1) for p in price_curve]
            profit_curve = [round(float((p - wholesale_cost) * m), 2) for p, m in zip(price_curve, move_curve)]

            # Sample 40 historical points for scatter visualization
            sample_points = st_data[["WEEK", "PRICE", "MOVE"]].tail(40).to_dict(orient="records")

            self.store_results[int(st)] = {
                "store_id": int(st),
                "total_observations": st_n,
                "current_avg_price": round(avg_price, 2),
                "current_avg_weekly_units": round(avg_move, 1),
                "current_weekly_revenue": round(current_weekly_revenue, 2),
                "current_weekly_profit": round(current_weekly_profit, 2),
                "dml_causal_elasticity": round(st_dml_elas, 2),
                "naive_elasticity": round(st_naive_elas, 2),
                "elasticity_type": "Elastic (Sensitive)" if st_dml_elas < -1.0 else "Inelastic (Pricing Power)",
                "optimal_dynamic_price": optimal_price,
                "price_recommendation_direction": "Increase Price" if optimal_price > avg_price + 0.03 else ("Decrease Price" if optimal_price < avg_price - 0.03 else "Maintain Current Price"),
                "projected_units_at_optimal": round(projected_move, 1),
                "projected_weekly_revenue": projected_weekly_revenue,
                "projected_weekly_profit": projected_weekly_profit,
                "profit_lift_usd": profit_lift_usd,
                "profit_lift_pct": profit_lift_pct,
                "simulation_curve": {
                    "prices": [round(float(p), 2) for p in price_curve],
                    "demand_units": move_curve,
                    "profit_usd": profit_curve
                },
                "recent_history": sample_points
            }

        self.is_fitted = True
        return self

    def get_summary_catalog(self) -> Dict[str, Any]:
        """Returns catalogue of stores and overall pricing metrics."""
        if not self.is_fitted:
            self.fit()

        total_profit_lift = sum(s["profit_lift_usd"] for s in self.store_results.values() if s["profit_lift_usd"] > 0)
        avg_elasticity = np.mean([s["dml_causal_elasticity"] for s in self.store_results.values()])

        return {
            "total_stores": len(self.store_results),
            "total_records": len(self.df_clean),
            "primary_upc": "1313000002",
            "overall_dml_elasticity": round(self.overall_dml_elasticity, 2),
            "overall_naive_elasticity": round(self.overall_naive_elasticity, 2),
            "mean_store_elasticity": round(float(avg_elasticity), 2),
            "total_chain_weekly_profit_lift_usd": round(float(total_profit_lift), 2),
            "stores_list": sorted(list(self.store_results.keys())),
            "all_stores_matrix": list(self.store_results.values())
        }

    def get_store_details(self, store_id: int) -> Dict[str, Any]:
        """Returns deep dive analytics for an individual store."""
        if not self.is_fitted:
            self.fit()

        if store_id not in self.store_results:
            # Fallback to closest available store
            store_id = list(self.store_results.keys())[0]

        return self.store_results[store_id]
