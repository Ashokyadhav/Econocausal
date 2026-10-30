"""
============================================================
ECONOCAUSAL: DOUBLE MACHINE LEARNING ENGINE
============================================================

Project:
Dynamic Pricing via Double Machine Learning

Methodology:
Partially Linear Double Machine Learning
Neyman Orthogonalization
K-Fold Cross-Fitting

Treatment:
PRICE (Continuous)

Outcome:
MOVE (Product Demand)

Developed for:
EconoCausal Dynamic Pricing Project
============================================================
"""

import numpy as np
import pandas as pd

from typing import List, Dict, Any

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scipy import stats

from sklearn.linear_model import Ridge


class DoubleMachineLearningEngine:

    """
    Continuous Treatment Double Machine Learning Engine.

    Treatment:
        PRICE

    Outcome:
        MOVE

    Estimates:
        1. Average Treatment Effect (ATE)
        2. Conditional Average Treatment Effect (CATE)
        3. Confidence Intervals
        4. Statistical Significance
        5. Model Diagnostics

    Uses:
        K-Fold Cross-Fitting
        Neyman Orthogonal Scores
        Partially Linear Regression
    """

    # --------------------------------------------------------
    # 1. INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        n_folds=5,
        random_seed=42,
        l2_reg=1.0
    ):

        self.n_folds = n_folds
        self.random_seed = random_seed
        self.l2_reg = l2_reg

        self.is_fitted = False

        self.feature_names = []
        self.encoded_feature_names = []

        self.ate_results = {}
        self.diagnostics = {}

        self.cate_model = None
        self.preprocessor = None

        self.feature_matrix_columns = None

    # --------------------------------------------------------
    # 2. DATA VALIDATION
    # --------------------------------------------------------

    def validate_data(
        self,
        df,
        feature_cols,
        outcome_col="MOVE",
        treatment_col="PRICE"
    ):

        required = (
            feature_cols +
            [outcome_col, treatment_col]
        )

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        if len(df) <= self.n_folds:

            raise ValueError(
                "Dataset must contain more rows than folds."
            )

        if not df[required].notna().all().all():

            raise ValueError(
                "Missing values found. Handle missing "
                "values before fitting the DML engine."
            )

        if not np.isfinite(
            df[[outcome_col, treatment_col]]
            .to_numpy(dtype=float)
        ).all():

            raise ValueError(
                "Outcome and treatment contain "
                "infinite or invalid values."
            )

        if not pd.api.types.is_numeric_dtype(
            df[outcome_col]
        ):

            raise ValueError(
                "Outcome must be numeric."
            )

        if not pd.api.types.is_numeric_dtype(
            df[treatment_col]
        ):

            raise ValueError(
                "Treatment must be numeric."
            )

        if df[treatment_col].nunique() < 2:

            raise ValueError(
                "Treatment must contain variation."
            )

    # --------------------------------------------------------
    # 3. FEATURE PREPROCESSING
    # --------------------------------------------------------

    def prepare_features(
        self,
        df,
        feature_cols
    ):

        X = df[feature_cols].copy()

        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numerical_cols = [
            col for col in feature_cols
            if col not in categorical_cols
        ]

        transformers = []

        if numerical_cols:

            numerical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ])

            transformers.append(
                (
                    "numeric",
                    numerical_pipeline,
                    numerical_cols
                )
            )

        if categorical_cols:

            categorical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )

            ])

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_cols
                )
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return X

    # --------------------------------------------------------
    # 4. FIT DML ENGINE
    # --------------------------------------------------------

    def fit(
        self,
        df,
        feature_cols,
        outcome_col="MOVE",
        treatment_col="PRICE"
    ):

        self.validate_data(
            df,
            feature_cols,
            outcome_col,
            treatment_col
        )

        self.feature_names = feature_cols.copy()

        X_raw = self.prepare_features(
            df,
            feature_cols
        )

        Y = df[outcome_col].to_numpy(dtype=float)

        T = df[treatment_col].to_numpy(dtype=float)

        n_samples = len(Y)

        # ----------------------------------------------------
        # Cross-fitting
        # ----------------------------------------------------

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        q_hat = np.zeros(n_samples)

        m_hat = np.zeros(n_samples)

        X_transformed = np.zeros(
            (n_samples, 0)
        )

        # Nuisance models

        outcome_model = ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=self.random_seed,
            n_jobs=-1
        )

        treatment_model = ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=self.random_seed + 1,
            n_jobs=-1
        )

        # ----------------------------------------------------
        # Stage 1: Cross-fitting
        # ----------------------------------------------------

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X_raw)
        ):

            X_train_raw = X_raw.iloc[train_idx]

            X_val_raw = X_raw.iloc[val_idx]

            Y_train = Y[train_idx]

            T_train = T[train_idx]

            # Fit preprocessing only on training fold

            fold_preprocessor = clone(
                self.preprocessor
            )

            X_train = fold_preprocessor.fit_transform(
                X_train_raw
            )

            X_val = fold_preprocessor.transform(
                X_val_raw
            )

            if fold == 0:

                X_transformed = np.zeros(
                    (
                        n_samples,
                        X_val.shape[1]
                    )
                )

                self.encoded_feature_names = (
                    fold_preprocessor.get_feature_names_out()
                )

            X_transformed[val_idx] = X_val

            # Outcome nuisance model

            q_model = clone(outcome_model)

            q_model.fit(
                X_train,
                Y_train
            )

            q_hat[val_idx] = q_model.predict(
                X_val
            )

            # Treatment nuisance model

            m_model = clone(treatment_model)

            m_model.fit(
                X_train,
                T_train
            )

            m_hat[val_idx] = m_model.predict(
                X_val
            )

        # ----------------------------------------------------
        # Stage 2: Orthogonal Residuals
        # ----------------------------------------------------

        Y_residual = Y - q_hat

        T_residual = T - m_hat

        residual_variance = np.var(
            T_residual,
            ddof=1
        )

        if residual_variance < 1e-12:

            raise ValueError(
                "Insufficient residual treatment variation. "
                "Check treatment overlap and feature selection."
            )

        # ----------------------------------------------------
        # Stage 3: Average Treatment Effect
        # ----------------------------------------------------

        denominator = np.sum(
            T_residual ** 2
        )

        ate = np.sum(
            T_residual * Y_residual
        ) / denominator

        ate_residuals = (
            Y_residual -
            ate * T_residual
        )

        # HC3 robust standard error

        leverage = (
            T_residual ** 2 /
            denominator
        )

        leverage = np.clip(
            leverage,
            0,
            0.95
        )

        hc3_weights = (
            ate_residuals /
            (1 - leverage)
        ) ** 2

        variance = np.sum(
            T_residual ** 2 *
            hc3_weights
        ) / denominator ** 2

        standard_error = np.sqrt(
            max(variance, 0)
        )

        z_score = (
            ate / standard_error
            if standard_error > 1e-12
            else np.nan
        )

        p_value = (
            2 * stats.norm.sf(abs(z_score))
            if np.isfinite(z_score)
            else np.nan
        )

        ci_lower = (
            ate - 1.96 * standard_error
        )

        ci_upper = (
            ate + 1.96 * standard_error
        )

        self.ate_results = {

            "ate": float(ate),

            "std_error": float(standard_error),

            "z_score": float(z_score),

            "p_value": float(p_value),

            "ci_lower": float(ci_lower),

            "ci_upper": float(ci_upper)

        }

        # ----------------------------------------------------
        # Stage 4: Heterogeneous Treatment Effects
        # ----------------------------------------------------

        # Exploratory linear CATE model:
        # tau(X) = beta_0 + beta_1*X_1 + ...

        cate_design = np.column_stack([

            T_residual,

            T_residual[:, None] *
            X_transformed

        ])

        self.cate_model = Ridge(
            alpha=self.l2_reg,
            fit_intercept=False
        )

        self.cate_model.fit(
            cate_design,
            Y_residual
        )

        # ----------------------------------------------------
        # Stage 5: Diagnostics
        # ----------------------------------------------------

        self.diagnostics = {

            "n_samples": int(n_samples),

            "n_features": int(
                X_transformed.shape[1]
            ),

            "n_folds": self.n_folds,

            "mean_price": float(np.mean(T)),

            "std_price": float(np.std(T)),

            "mean_demand": float(np.mean(Y)),

            "treatment_residual_std": float(
                np.std(T_residual)
            ),

            "outcome_residual_std": float(
                np.std(Y_residual)
            ),

            "outcome_nuisance_r2": float(
                1 -
                np.sum(Y_residual ** 2) /
                np.sum(
                    (Y - np.mean(Y)) ** 2
                )
            ),

            "treatment_residual_correlation": float(
                np.corrcoef(
                    T_residual,
                    Y_residual
                )[0, 1]
            )

        }

        self.is_fitted = True

        return self

    # --------------------------------------------------------
    # 5. PREDICT CATE
    # --------------------------------------------------------

    def predict_cate(
        self,
        df
    ):

        if not self.is_fitted:

            raise ValueError(
                "DML engine has not been fitted."
            )

        X_raw = self.prepare_prediction_features(
            df
        )

        X_scaled = self.preprocessor.transform(
            X_raw
        )

        # Reconstruct fitted CATE design

        # The fitted Ridge coefficients contain:
        # 1. Intercept treatment effect
        # 2. Feature-specific treatment effects

        coefficients = self.cate_model.coef_

        intercept_effect = coefficients[0]

        feature_effects = coefficients[1:]

        cate = (
            intercept_effect +
            X_scaled @ feature_effects
        )

        return {

            "cate": cate,

            "mean_cate": float(
                np.mean(cate)
            )

        }

    # --------------------------------------------------------
    # 6. PREDICTION FEATURE PREPARATION
    # --------------------------------------------------------

    def prepare_prediction_features(
        self,
        df
    ):

        missing = [
            col for col in self.feature_names
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing prediction features: {missing}"
            )

        return df[self.feature_names].copy()

    # --------------------------------------------------------
    # 7. SUMMARY METRICS
    # --------------------------------------------------------

    def get_summary_metrics(self):

        if not self.is_fitted:

            raise ValueError(
                "Fit the model before requesting metrics."
            )

        return {

            "ATE Results": self.ate_results,

            "Diagnostics": self.diagnostics,

            "Treatment": "PRICE",

            "Outcome": "MOVE",

            "Method": "Partially Linear Double Machine Learning"

        }
