"""
============================================================
ECONOCAUSAL: DOUBLE MACHINE LEARNING ENGINE
============================================================

Project: Dynamic Pricing via Double Machine Learning

Treatment : log(PRICE)
Outcome   : log(MOVE)

Methodology:
1. Partially Linear Double Machine Learning
2. K-Fold Cross-Fitting
3. Neyman Orthogonalization
4. Average Treatment Effect (ATE)
5. Conditional Average Treatment Effect (CATE)
6. Statistical Inference
7. Model Diagnostics

============================================================
"""

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


class DoubleMachineLearningEngine:

    """
    Continuous Treatment Double Machine Learning Engine.

    Treatment:
        log(PRICE)

    Outcome:
        log(MOVE)

    Features:
        Store, product and time-related variables.

    Estimation:
        Partially Linear DML.
    """

    # ========================================================
    # 1. INITIALIZATION
    # ========================================================

    def __init__(
        self,
        n_folds=5,
        random_seed=42,
        cate_alpha=1.0,
        nuisance_model="extra_trees",
        n_estimators=200,
        min_samples_leaf=5
    ):

        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

        if nuisance_model not in [
            "extra_trees",
            "random_forest"
        ]:
            raise ValueError(
                "Invalid nuisance model."
            )

        self.n_folds = n_folds
        self.random_seed = random_seed
        self.cate_alpha = cate_alpha

        self.nuisance_model = nuisance_model
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

        self.feature_names = []

        self.preprocessor = None
        self.cate_model = None

        self.is_fitted = False

        self.ate_results = {}
        self.diagnostics = {}

    # ========================================================
    # 2. ONE HOT ENCODER COMPATIBILITY
    # ========================================================

    @staticmethod
    def _make_encoder():

        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

    # ========================================================
    # 3. CREATE PREPROCESSOR
    # ========================================================

    def _make_preprocessor(self, X):

        numeric_cols = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_cols = [
            col for col in X.columns
            if col not in numeric_cols
        ]

        transformers = []

        if numeric_cols:

            numeric_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ])

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_cols
                )
            )

        if categorical_cols:

            categorical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    self._make_encoder()
                )

            ])

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_cols
                )
            )

        if not transformers:

            raise ValueError(
                "No usable features found."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

    # ========================================================
    # 4. CREATE NUISANCE MODEL
    # ========================================================

    def _make_regressor(self, seed_offset=0):

        params = {

            "n_estimators": self.n_estimators,

            "min_samples_leaf": self.min_samples_leaf,

            "random_state": (
                self.random_seed + seed_offset
            ),

            "n_jobs": -1

        }

        if self.nuisance_model == "random_forest":

            return RandomForestRegressor(**params)

        return ExtraTreesRegressor(**params)

    # ========================================================
    # 5. INPUT VALIDATION
    # ========================================================

    def validate_data(
        self,
        df,
        feature_cols,
        outcome_col,
        treatment_col
    ):

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if not feature_cols:

            raise ValueError(
                "Feature columns cannot be empty."
            )

        required = list(dict.fromkeys(
            feature_cols + [
                outcome_col,
                treatment_col
            ]
        ))

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        if len(df) < self.n_folds * 2:

            raise ValueError(
                "Insufficient observations for cross-fitting."
            )

        for col in [outcome_col, treatment_col]:

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            if not np.isfinite(values).all():

                raise ValueError(
                    f"{col} contains missing or invalid values."
                )

            if np.unique(values).size < 2:

                raise ValueError(
                    f"{col} must contain variation."
                )

    # ========================================================
    # 6. FIT DML MODEL
    # ========================================================

    def fit(
        self,
        df,
        feature_cols,
        outcome_col="log_move",
        treatment_col="log_price"
    ):

        self.validate_data(
            df,
            feature_cols,
            outcome_col,
            treatment_col
        )

        self.feature_names = list(feature_cols)

        X = df[self.feature_names].copy().reset_index(
            drop=True
        )

        Y = pd.to_numeric(
            df[outcome_col]
        ).to_numpy(dtype=float)

        T = pd.to_numeric(
            df[treatment_col]
        ).to_numpy(dtype=float)

        n = len(Y)

        # ----------------------------------------------------
        # Initialize cross-fitting
        # ----------------------------------------------------

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        q_hat = np.zeros(n)

        m_hat = np.zeros(n)

        X_oof = None

        template_preprocessor = (
            self._make_preprocessor(X)
        )

        # ----------------------------------------------------
        # Stage 1: Nuisance estimation
        # ----------------------------------------------------

        for fold, (train_idx, valid_idx) in enumerate(
            kf.split(X)
        ):

            X_train_raw = X.iloc[train_idx]

            X_valid_raw = X.iloc[valid_idx]

            Y_train = Y[train_idx]

            T_train = T[train_idx]

            # Fit preprocessing only on training data

            fold_preprocessor = clone(
                template_preprocessor
            )

            X_train = fold_preprocessor.fit_transform(
                X_train_raw
            )

            X_valid = fold_preprocessor.transform(
                X_valid_raw
            )

            if X_oof is None:

                X_oof = np.zeros(
                    (n, X_valid.shape[1])
                )

            X_oof[valid_idx] = X_valid

            # Outcome model

            model_y = self._make_regressor(
                seed_offset=fold
            )

            model_y.fit(
                X_train,
                Y_train
            )

            q_hat[valid_idx] = model_y.predict(
                X_valid
            )

            # Treatment model

            model_t = self._make_regressor(
                seed_offset=100 + fold
            )

            model_t.fit(
                X_train,
                T_train
            )

            m_hat[valid_idx] = model_t.predict(
                X_valid
            )

        # ----------------------------------------------------
        # Stage 2: Orthogonal residualization
        # ----------------------------------------------------

        Y_residual = Y - q_hat

        T_residual = T - m_hat

        denominator = np.sum(
            T_residual ** 2
        )

        if denominator <= 1e-12:

            raise ValueError(
                "Insufficient residual treatment variation."
            )

        # ----------------------------------------------------
        # Stage 3: Average Treatment Effect
        # ----------------------------------------------------

        ate = np.sum(
            T_residual * Y_residual
        ) / denominator

        residuals = (
            Y_residual -
            ate * T_residual
        )

        # HC3-style heteroskedasticity-robust SE

        leverage = (
            T_residual ** 2 /
            denominator
        )

        leverage = np.clip(
            leverage,
            0,
            0.95
        )

        adjusted_residuals = (
            residuals /
            (1 - leverage)
        )

        variance = np.sum(
            T_residual ** 2 *
            adjusted_residuals ** 2
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

        critical_value = stats.norm.ppf(0.975)

        ci_lower = (
            ate -
            critical_value * standard_error
        )

        ci_upper = (
            ate +
            critical_value * standard_error
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
        # Stage 4: Exploratory CATE estimation
        # ----------------------------------------------------

        cate_design = np.column_stack([

            T_residual,

            T_residual[:, None] *
            X_oof

        ])

        self.cate_model = Ridge(
            alpha=self.cate_alpha,
            fit_intercept=False
        )

        self.cate_model.fit(
            cate_design,
            Y_residual
        )

        # ----------------------------------------------------
        # Stage 5: Fit final preprocessing model
        # ----------------------------------------------------

        self.preprocessor = clone(
            template_preprocessor
        )

        X_all = self.preprocessor.fit_transform(X)

        # ----------------------------------------------------
        # Stage 6: Model diagnostics
        # ----------------------------------------------------

        outcome_r2 = (
            1 -
            np.sum(Y_residual ** 2) /
            np.sum(
                (Y - np.mean(Y)) ** 2
            )
        )

        treatment_r2 = (
            1 -
            np.sum(T_residual ** 2) /
            np.sum(
                (T - np.mean(T)) ** 2
            )
        )

        self.diagnostics = {

            "sample_size": int(n),

            "number_of_features": int(
                X_all.shape[1]
            ),

            "cross_folds": self.n_folds,

            "outcome_oof_r2": float(
                outcome_r2
            ),

            "treatment_oof_r2": float(
                treatment_r2
            ),

            "treatment_residual_std": float(
                np.std(T_residual, ddof=1)
            ),

            "outcome_residual_std": float(
                np.std(Y_residual, ddof=1)
            )

        }

        self.is_fitted = True

        return self

    # ========================================================
    # 7. PREDICT CATE
    # ========================================================

    def predict_cate(self, df):

        if not self.is_fitted:

            raise ValueError(
                "Fit the DML model first."
            )

        missing = [
            col for col in self.feature_names
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing prediction features: {missing}"
            )

        X = df[self.feature_names].copy()

        X_transformed = (
            self.preprocessor.transform(X)
        )

        coefficients = self.cate_model.coef_

        intercept_effect = coefficients[0]

        feature_effects = coefficients[1:]

        cate = (
            intercept_effect +
            X_transformed @ feature_effects
        )

        return cate

    # ========================================================
    # 8. GET SUMMARY
    # ========================================================

    def get_summary_metrics(self):

        if not self.is_fitted:

            raise ValueError(
                "Model has not been fitted."
            )

        return {

            "method":
                "Partially Linear Double Machine Learning",

            "treatment":
                "Continuous Price",

            "outcome":
                "Product Demand",

            "ATE":
                self.ate_results,

            "Diagnostics":
                self.diagnostics,

            "CATE_note":
                "Exploratory linear heterogeneous treatment estimates."

        }
