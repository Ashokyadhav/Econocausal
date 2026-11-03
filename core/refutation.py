"""
EconoCausal - Causal Refutation and Sensitivity Suite

Implements diagnostic tests for Double Machine Learning:

1. Placebo Treatment Test
2. Random Noise Invariance Test
3. Data Subset Stability Test
4. Robustness Value Availability Check

Treatment : log(PRICE)
Outcome   : log(MOVE)

Note:
The robustness value is not calculated using an invented
approximation. A valid Cinelli-Hazlett calculation requires
additional partial R-squared sensitivity information.
"""

import numpy as np
import pandas as pd

from typing import Dict, Any, List

from .dml_engine import DoubleMachineLearningEngine


class CausalRefutationSuite:
    """
    Runs diagnostic tests to examine the stability of
    causal estimates produced by DoubleMachineLearningEngine.
    """

    def __init__(
        self,
        random_seed: int = 42,
        n_subsamples: int = 10,
        subset_fraction: float = 0.80,
        stability_threshold: float = 10.0,
        noise_threshold: float = 8.0
    ):

        if n_subsamples < 1:
            raise ValueError("n_subsamples must be at least 1.")

        if not 0 < subset_fraction < 1:
            raise ValueError("subset_fraction must be between 0 and 1.")

        self.random_seed = random_seed
        self.n_subsamples = n_subsamples
        self.subset_fraction = subset_fraction
        self.stability_threshold = stability_threshold
        self.noise_threshold = noise_threshold

        self.rng = np.random.default_rng(random_seed)

    # ---------------------------------------------------------
    # Helper: Create a fresh DML engine
    # ---------------------------------------------------------

    def _create_engine(self, base_engine, seed_offset=0):

        return DoubleMachineLearningEngine(
            n_folds=base_engine.n_folds,
            random_seed=base_engine.random_seed + seed_offset,
            cate_alpha=base_engine.cate_alpha,
            nuisance_model=base_engine.nuisance_model,
            n_estimators=base_engine.n_estimators,
            min_samples_leaf=base_engine.min_samples_leaf
        )

    # ---------------------------------------------------------
    # Helper: Extract ATE
    # ---------------------------------------------------------

    @staticmethod
    def _extract_ate(engine):

        if not engine.ate_results:
            raise ValueError("ATE results are unavailable.")

        required_keys = ["ate", "std_error", "p_value"]

        for key in required_keys:
            if key not in engine.ate_results:
                raise ValueError(
                    f"Missing '{key}' in DML ATE results."
                )

        return {
            "ate": float(engine.ate_results["ate"]),
            "std_error": float(engine.ate_results["std_error"]),
            "p_value": float(engine.ate_results["p_value"])
        }

    # ---------------------------------------------------------
    # Helper: Percentage change
    # ---------------------------------------------------------

    @staticmethod
    def _percentage_change(original, updated):

        denominator = max(abs(original), 1e-8)

        return abs(updated - original) / denominator * 100.0

    # ---------------------------------------------------------
    # Main Refutation Function
    # ---------------------------------------------------------

    def run_all_refutations(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        base_engine: DoubleMachineLearningEngine,
        outcome_col: str = "log_move",
        treatment_col: str = "log_price"
    ) -> Dict[str, Any]:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if not base_engine.is_fitted:
            raise ValueError(
                "The base DML engine must be fitted before "
                "running refutation tests."
            )

        if treatment_col not in df.columns:
            raise ValueError(
                f"Treatment column '{treatment_col}' not found."
            )

        if outcome_col not in df.columns:
            raise ValueError(
                f"Outcome column '{outcome_col}' not found."
            )

        if not feature_cols:
            raise ValueError("Feature columns cannot be empty.")

        missing_features = [
            col for col in feature_cols
            if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}"
            )

        if len(df) < base_engine.n_folds * 2:
            raise ValueError(
                "Insufficient observations for refutation tests."
            )

        results = {}

        # Original baseline estimate
        original = self._extract_ate(base_engine)

        original_ate = original["ate"]

        # =====================================================
        # 1. PLACEBO TREATMENT TEST
        # =====================================================

        df_placebo = df.copy()

        # Randomly shuffle treatment assignments
        df_placebo[treatment_col] = self.rng.permutation(
            df_placebo[treatment_col].to_numpy()
        )

        placebo_engine = self._create_engine(
            base_engine,
            seed_offset=10
        )

        placebo_engine.fit(
            df_placebo,
            feature_cols,
            outcome_col=outcome_col,
            treatment_col=treatment_col
        )

        placebo = self._extract_ate(placebo_engine)

        placebo_change = self._percentage_change(
            original_ate,
            placebo["ate"]
        )

        # A nonsignificant placebo is a useful diagnostic,
        # but does not prove that the original effect is causal.
        placebo_passed = (
            placebo["p_value"] > 0.05
            and placebo_change > 0
        )

        results["placebo_treatment"] = {

            "name": "Placebo Treatment Test",

            "description": (
                "Randomly shuffles treatment values to examine "
                "whether the estimated relationship persists."
            ),

            "original_effect": round(original_ate, 6),

            "placebo_effect": round(placebo["ate"], 6),

            "placebo_p_value": round(
                placebo["p_value"], 6
            ),

            "percentage_change": round(
                placebo_change, 2
            ),

            "passed": bool(placebo_passed),

            "status": (
                "PASSED" if placebo_passed else "FAILED"
            ),

            "interpretation": (
                "The shuffled-treatment estimate is statistically "
                "insignificant."
                if placebo_passed
                else
                "The shuffled-treatment estimate remains "
                "statistically significant or the diagnostic "
                "did not meet its criterion."
            )
        }

        # =====================================================
        # 2. RANDOM NOISE INVARIANCE TEST
        # =====================================================

        df_noise = df.copy()

        noise_column = "_random_noise_variable"

        df_noise[noise_column] = self.rng.normal(
            loc=0,
            scale=1,
            size=len(df_noise)
        )

        noise_features = feature_cols + [noise_column]

        noise_engine = self._create_engine(
            base_engine,
            seed_offset=20
        )

        noise_engine.fit(
            df_noise,
            noise_features,
            outcome_col=outcome_col,
            treatment_col=treatment_col
        )

        noise_result = self._extract_ate(noise_engine)

        noise_change = self._percentage_change(
            original_ate,
            noise_result["ate"]
        )

        noise_passed = (
            noise_change < self.noise_threshold
        )

        results["random_noise_invariance"] = {

            "name": "Random Noise Invariance Test",

            "description": (
                "Adds an independent random variable to the "
                "feature set and checks the change in ATE."
            ),

            "original_effect": round(
                original_ate, 6
            ),

            "adjusted_effect": round(
                noise_result["ate"], 6
            ),

            "percentage_change": round(
                noise_change, 2
            ),

            "threshold": self.noise_threshold,

            "passed": bool(noise_passed),

            "status": (
                "PASSED" if noise_passed else "FAILED"
            ),

            "interpretation": (
                "The estimate remained relatively stable "
                "after adding an independent random variable."
                if noise_passed
                else
                "The estimate changed beyond the specified "
                "stability threshold."
            )
        }

        # =====================================================
        # 3. DATA SUBSET STABILITY TEST
        # =====================================================

        subset_effects = []

        subset_size = int(
            len(df) * self.subset_fraction
        )

        for iteration in range(self.n_subsamples):

            subset_indices = self.rng.choice(
                len(df),
                size=subset_size,
                replace=False
            )

            df_subset = (
                df.iloc[subset_indices]
                .reset_index(drop=True)
            )

            subset_engine = self._create_engine(
                base_engine,
                seed_offset=30 + iteration
            )

            subset_engine.fit(
                df_subset,
                feature_cols,
                outcome_col=outcome_col,
                treatment_col=treatment_col
            )

            subset_result = self._extract_ate(
                subset_engine
            )

            subset_effects.append(
                subset_result["ate"]
            )

        subset_effects = np.asarray(
            subset_effects,
            dtype=float
        )

        median_subset_effect = float(
            np.median(subset_effects)
        )

        subset_change = self._percentage_change(
            original_ate,
            median_subset_effect
        )

        subset_std = float(
            np.std(subset_effects, ddof=1)
            if len(subset_effects) > 1
            else 0.0
        )

        subset_passed = (
            subset_change < self.stability_threshold
        )

        results["subset_stability"] = {

            "name": "Data Subset Stability Test",

            "description": (
                f"Repeatedly estimates the treatment effect "
                f"using {self.subset_fraction:.0%} random subsets."
            ),

            "original_effect": round(
                original_ate, 6
            ),

            "median_subset_effect": round(
                median_subset_effect, 6
            ),

            "subset_effect_std": round(
                subset_std, 6
            ),

            "minimum_subset_effect": round(
                float(np.min(subset_effects)), 6
            ),

            "maximum_subset_effect": round(
                float(np.max(subset_effects)), 6
            ),

            "number_of_subsamples": self.n_subsamples,

            "percentage_change": round(
                subset_change, 2
            ),

            "threshold": self.stability_threshold,

            "passed": bool(subset_passed),

            "status": (
                "PASSED" if subset_passed else "FAILED"
            ),

            "interpretation": (
                "The median treatment effect remained within "
                "the specified stability threshold."
                if subset_passed
                else
                "The estimated treatment effect varied beyond "
                "the specified stability threshold."
            )
        }

        # =====================================================
        # 4. ROBUSTNESS VALUE AVAILABILITY
        # =====================================================

        # A valid Cinelli-Hazlett robustness value cannot be
        # calculated from the DML ATE and standard error alone.
        # It requires suitable partial R-squared sensitivity
        # information for the treatment and outcome models.

        results["robustness_value"] = {

            "name": "Cinelli-Hazlett Robustness Value",

            "status": "NOT_COMPUTED",

            "robustness_value_pct": None,

            "description": (
                "Sensitivity analysis for unobserved confounding."
            ),

            "reason": (
                "The current DML engine does not provide the "
                "partial R-squared quantities required for a "
                "valid Cinelli-Hazlett robustness calculation."
            ),

            "interpretation": (
                "Additional sensitivity analysis is required. "
                "No numerical robustness value is reported."
            )
        }

        # =====================================================
        # 5. SUMMARY
        # =====================================================

        passed_tests = sum(
            1
            for key, value in results.items()
            if key != "robustness_value"
            and value.get("passed", False)
        )

        total_tests = 3

        results["summary"] = {

            "total_diagnostic_tests": total_tests,

            "passed_tests": passed_tests,

            "failed_tests": total_tests - passed_tests,

            "overall_status": (
                "DIAGNOSTICS_PASSED"
                if passed_tests == total_tests
                else "REVIEW_REQUIRED"
            ),

            "warning": (
                "Passing refutation tests does not establish "
                "causality. Identification assumptions, "
                "confounding, treatment overlap and panel "
                "dependence must also be evaluated."
            )
        }

        return results
