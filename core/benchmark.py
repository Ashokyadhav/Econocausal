python
"""
EconoCausal - Model Benchmark & Uplift Evaluator

Compares Double Machine Learning (DML) against:
1. Traditional Correlational Predictive ML (Churn Risk Targeting)
2. Naive Observational Difference-in-Means
3. T-Learner (Two Separate Models)

Computes:
- ATE comparisons
- Cumulative uplift curves
- Targeting efficiency metrics
- Qini-style cumulative ground-truth uplift curves for synthetic data
"""

import numpy as np
import pandas as pd

from typing import Dict, Any, List

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ModelComparator:
    """
    Evaluates causal AI (DML) against correlational ML baselines.

    The evaluator compares:
    - DML-based causal targeting
    - Churn-risk targeting
    - Random targeting
    - T-Learner treatment-effect estimates
    - Naive observational treatment-effect estimates
    """

    REQUIRED_COLUMNS = {
        "converted",
        "discount_offered",
    }

    GROUND_TRUTH_COLUMNS = {
        "true_tau_10",
        "true_tau_20",
        "archetype",
    }

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()

        self.t_learner_m0 = None
        self.t_learner_m20 = None
        self.churn_classifier = None

        self.feature_cols = None
        self.is_fitted = False

    def _validate_columns(
        self,
        df: pd.DataFrame,
        required_columns: set,
    ) -> None:
        """Raise a clear error when required columns are missing."""

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def _validate_treatments(
        self,
        treatment: np.ndarray,
        required_treatments: List[int],
    ) -> None:
        """Ensure that all required discount groups contain observations."""

        for value in required_treatments:
            if not np.any(treatment == value):
                raise ValueError(
                    f"No observations found for discount treatment {value}."
                )

    def fit_baselines(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> None:
        """
        Fit the churn classifier and T-Learner baseline models.

        The churn classifier predicts non-conversion.
        The T-Learner fits separate outcome models for:
        - Control customers (discount = 0)
        - Customers receiving a $20 discount
        """

        if df.empty:
            raise ValueError("Training dataset cannot be empty.")

        self._validate_columns(
            df,
            self.REQUIRED_COLUMNS.union(feature_cols),
        )

        self._validate_treatments(
            df["discount_offered"].to_numpy(),
            [0, 20],
        )

        if df[feature_cols].isnull().any().any():
            raise ValueError(
                "Feature columns contain missing values. "
                "Clean or impute them before fitting."
            )

        if df["converted"].isnull().any():
            raise ValueError(
                "The converted column contains missing values."
            )

        X = df[feature_cols].to_numpy(dtype=float)
        Y = df["converted"].to_numpy()
        T = df["discount_offered"].to_numpy()

        if not np.isin(Y, [0, 1]).all():
            raise ValueError(
                "The converted column must contain only 0 and 1."
            )

        X_scaled = self.scaler.fit_transform(X)

        # 1. Traditional churn / non-conversion classifier.
        self.churn_classifier = LogisticRegression(
            max_iter=500,
            random_state=self.random_seed,
        )

        self.churn_classifier.fit(X_scaled, 1 - Y)

        # 2. T-Learner: separate outcome models.
        idx_t0 = T == 0
        idx_t20 = T == 20

        self.t_learner_m0 = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=self.random_seed,
        )

        self.t_learner_m0.fit(
            X_scaled[idx_t0],
            Y[idx_t0],
        )

        self.t_learner_m20 = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=self.random_seed,
        )

        self.t_learner_m20.fit(
            X_scaled[idx_t20],
            Y[idx_t20],
        )

        self.feature_cols = list(feature_cols)
        self.is_fitted = True

    def evaluate_all(
        self,
        df: pd.DataFrame,
        dml_engine,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Compare DML, churn-based, T-Learner, naive and random targeting.

        When synthetic ground-truth columns are available, use them to
        evaluate the actual treatment effects among targeted customers.

        Otherwise, use the estimated DML treatment effects and clearly
        label them as estimates rather than known causal ground truth.
        """

        if df.empty:
            raise ValueError("Evaluation dataset cannot be empty.")

        if not self.is_fitted:
            raise ValueError(
                "Baseline models have not been fitted. "
                "Call fit_baselines() before evaluate_all()."
            )

        if list(feature_cols) != self.feature_cols:
            raise ValueError(
                "feature_cols must match the columns used during fitting."
            )

        self._validate_columns(
            df,
            self.REQUIRED_COLUMNS.union(feature_cols),
        )

        if df[feature_cols].isnull().any().any():
            raise ValueError(
                "Evaluation features contain missing values."
            )

        if df["converted"].isnull().any():
            raise ValueError(
                "The converted column contains missing values."
            )

        if df["discount_offered"].isnull().any():
            raise ValueError(
                "The discount_offered column contains missing values."
            )

        X = df[feature_cols].to_numpy(dtype=float)
        Y = df["converted"].to_numpy(dtype=float)
        T = df["discount_offered"].to_numpy()

        if not np.isin(Y, [0, 1]).all():
            raise ValueError(
                "The converted column must contain only 0 and 1."
            )

        self._validate_treatments(T, [0, 10, 20])

        X_scaled = self.scaler.transform(X)

        # 1. Naive observational difference in means.
        mean_y_t0 = float(np.mean(Y[T == 0]))
        mean_y_t10 = float(np.mean(Y[T == 10]))
        mean_y_t20 = float(np.mean(Y[T == 20]))

        naive_ate_10 = mean_y_t10 - mean_y_t0
        naive_ate_20 = mean_y_t20 - mean_y_t0

        # 2. T-Learner predictions.
        pred_y0_tlearner = self.t_learner_m0.predict(X_scaled)
        pred_y20_tlearner = self.t_learner_m20.predict(X_scaled)

        tau_20_tlearner = pred_y20_tlearner - pred_y0_tlearner

        # 3. Traditional churn predictions.
        churn_risk_pred = self.churn_classifier.predict_proba(
            X_scaled
        )[:, 1]

        # 4. DML treatment-effect predictions.
        dml_preds = dml_engine.predict_cate(df)

        required_dml_predictions = {"tau_10", "tau_20"}

        if not required_dml_predictions.issubset(dml_preds.keys()):
            raise ValueError(
                "DML predictions must contain 'tau_10' and 'tau_20'."
            )

        tau_10_dml = np.asarray(dml_preds["tau_10"], dtype=float)
        tau_20_dml = np.asarray(dml_preds["tau_20"], dtype=float)

        if len(tau_10_dml) != len(df) or len(tau_20_dml) != len(df):
            raise ValueError(
                "DML prediction lengths must match the evaluation dataset."
            )

        if not (
            np.isfinite(tau_10_dml).all()
            and np.isfinite(tau_20_dml).all()
        ):
            raise ValueError(
                "DML predictions contain non-finite values."
            )

        # 5. Ground-truth availability.
        has_ground_truth = self.GROUND_TRUTH_COLUMNS.issubset(
            df.columns
        )

        true_ate_10 = None
        true_ate_20 = None

        if has_ground_truth:
            true_tau_10 = pd.to_numeric(
                df["true_tau_10"],
                errors="coerce",
            ).to_numpy(dtype=float)

            true_tau_20 = pd.to_numeric(
                df["true_tau_20"],
                errors="coerce",
            ).to_numpy(dtype=float)

            if not (
                np.isfinite(true_tau_10).all()
                and np.isfinite(true_tau_20).all()
            ):
                raise ValueError(
                    "Ground-truth treatment effects contain "
                    "missing or non-finite values."
                )

            true_ate_10 = float(np.mean(true_tau_10))
            true_ate_20 = float(np.mean(true_tau_20))

        ate_results = getattr(dml_engine, "ate_results", {})

        dml_ate_10 = ate_results.get(10, {}).get("ate", 0.0)
        dml_ate_20 = ate_results.get(20, {}).get("ate", 0.0)

        # 6. Targeting top 20% of customers.
        top_k_pct = 0.20
        n_top = max(1, int(len(df) * top_k_pct))

        dml_top_indices = np.argsort(tau_20_dml)[::-1][:n_top]

        churn_top_indices = np.argsort(
            churn_risk_pred
        )[::-1][:n_top]

        random_indices = np.random.default_rng(
            self.random_seed
        ).choice(
            len(df),
            size=n_top,
            replace=False,
        )

        if has_ground_truth:
            dml_targeted_lift = float(
                np.mean(true_tau_20[dml_top_indices])
            )

            churn_targeted_lift = float(
                np.mean(true_tau_20[churn_top_indices])
            )

            random_targeted_lift = float(
                np.mean(true_tau_20[random_indices])
            )

            archetypes = df["archetype"].astype(str).to_numpy()

            dml_lost_causes_pct = float(
                np.mean(archetypes[dml_top_indices] == "Lost Causes")
                * 100.0
            )

            churn_lost_causes_pct = float(
                np.mean(archetypes[churn_top_indices] == "Lost Causes")
                * 100.0
            )

            dml_persuadables_pct = float(
                np.mean(archetypes[dml_top_indices] == "Persuadables")
                * 100.0
            )

            churn_persuadables_pct = float(
                np.mean(archetypes[churn_top_indices] == "Persuadables")
                * 100.0
            )

        else:
            # These are estimated effects, not known ground truth.
            dml_targeted_lift = float(
                np.mean(tau_20_dml[dml_top_indices])
            )

            churn_targeted_lift = float(
                np.mean(tau_20_dml[churn_top_indices])
            )

            random_targeted_lift = float(
                np.mean(tau_20_dml[random_indices])
            )

            dml_lost_causes_pct = None
            churn_lost_causes_pct = None
            dml_persuadables_pct = None
            churn_persuadables_pct = None

        # 7. Cumulative uplift curves.
        # For synthetic data these use known effects, not conventional
        # observed-outcome Qini calculations.
        deciles = np.linspace(0.1, 1.0, 10)

        qini_dml = []
        qini_churn = []
        qini_random = []

        for d in deciles:
            k = max(1, int(len(df) * d))

            idx_dml = np.argsort(tau_20_dml)[::-1][:k]

            idx_churn = np.argsort(
                churn_risk_pred
            )[::-1][:k]

            if has_ground_truth:
                cum_dml = float(
                    np.sum(true_tau_20[idx_dml])
                )

                cum_churn = float(
                    np.sum(true_tau_20[idx_churn])
                )

                cum_random = float(
                    d * np.sum(true_tau_20)
                )

            else:
                cum_dml = float(
                    np.sum(tau_20_dml[idx_dml])
                )

                cum_churn = float(
                    np.sum(tau_20_dml[idx_churn])
                )

                cum_random = float(
                    d * np.sum(tau_20_dml)
                )

            qini_dml.append(round(cum_dml, 4))
            qini_churn.append(round(cum_churn, 4))
            qini_random.append(round(cum_random, 4))

        # 8. Generate conclusions from actual results.
        if has_ground_truth:
            if dml_persuadables_pct > churn_persuadables_pct:
                conclusion = (
                    f"DML selected more Persuadables "
                    f"({dml_persuadables_pct:.1f}%) than "
                    f"churn-based targeting "
                    f"({churn_persuadables_pct:.1f}%)."
                )
            elif dml_persuadables_pct < churn_persuadables_pct:
                conclusion = (
                    f"Churn-based targeting selected more "
                    f"Persuadables ({churn_persuadables_pct:.1f}%) "
                    f"than DML ({dml_persuadables_pct:.1f}%)."
                )
            else:
                conclusion = (
                    "DML and churn-based targeting selected "
                    "the same percentage of Persuadables."
                )
        else:
            conclusion = (
                "Ground-truth archetypes are unavailable. "
                "The targeting comparison uses estimated DML "
                "treatment effects, so the true causal lift "
                "cannot be directly verified."
            )

        # 9. Assemble results.
        return {
            "ate_comparison": {
                "naive_difference": {
                    "$10": round(naive_ate_10, 4),
                    "$20": round(naive_ate_20, 4),
                    "bias_note": (
                        "Observational differences may be biased "
                        "by confounding and non-random discount assignment."
                    ),
                },
                "t_learner": {
                    "$20": round(
                        float(np.mean(tau_20_tlearner)), 4
                    ),
                    "bias_note": (
                        "The T-Learner estimates treatment effects "
                        "using separate outcome models. Its accuracy "
                        "depends on model fit and treatment-group data."
                    ),
                },
                "double_machine_learning": {
                    "$10": round(float(dml_ate_10), 4),
                    "$20": round(float(dml_ate_20), 4),
                    "ci_95_d20": [
                        round(
                            float(
                                ate_results.get(20, {}).get(
                                    "ci_lower", 0.0
                                )
                            ),
                            4,
                        ),
                        round(
                            float(
                                ate_results.get(20, {}).get(
                                    "ci_upper", 0.0
                                )
                            ),
                            4,
                        ),
                    ],
                },
                "true_ground_truth": {
                    "$10": (
                        round(true_ate_10, 4)
                        if true_ate_10 is not None
                        else None
                    ),
                    "$20": (
                        round(true_ate_20, 4)
                        if true_ate_20 is not None
                        else None
                    ),
                },
            },
            "targeting_efficiency_top20pct": {
                "dml_causal_lift": round(
                    dml_targeted_lift * 100.0, 2
                ),
                "churn_targeted_lift": round(
                    churn_targeted_lift * 100.0, 2
                ),
                "random_lift": round(
                    random_targeted_lift * 100.0, 2
                ),
                "dml_persuadables_share_pct": (
                    round(dml_persuadables_pct, 2)
                    if dml_persuadables_pct is not None
                    else None
                ),
                "dml_lost_causes_share_pct": (
                    round(dml_lost_causes_pct, 2)
                    if dml_lost_causes_pct is not None
                    else None
                ),
                "churn_lost_causes_share_pct": (
                    round(churn_lost_causes_pct, 2)
                    if churn_lost_causes_pct is not None
                    else None
                ),
                "churn_persuadables_share_pct": (
                    round(churn_persuadables_pct, 2)
                    if churn_persuadables_pct is not None
                    else None
                ),
                "conclusion": conclusion,
            },
            "qini_curve": {
                "metric_description": (
                    "Cumulative ground-truth uplift for synthetic data; "
                    "estimated cumulative uplift when ground truth "
                    "is unavailable. This is not a conventional "
                    "observed-outcome Qini calculation."
                ),
                "deciles": [
                    int(d * 100) for d in deciles
                ],
                "dml_cumulative_lift": qini_dml,
                "churn_ml_cumulative_lift": qini_churn,
                "random_cumulative_lift": qini_random,
            },
        }
```
