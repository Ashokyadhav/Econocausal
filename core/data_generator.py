"""
EconoCausal - Observational Pricing Data Generator
Generates realistic e-commerce/SaaS customer datasets with confounded promotional assignments,
counterfactual outcomes, and known ground-truth Individual Treatment Effects (ITE).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class CustomerDataGenerator:
    """
    Simulates observational customer behavior with realistic confounding
    and heterogeneous treatment effects across 4 causal archetypes:
    - Persuadables (High causal uplift)
    - Sure Things (High baseline conversion, near-zero uplift)
    - Lost Causes (Low baseline conversion, near-zero uplift)
    - Sleeping Dogs (Negative uplift / annoyance)
    """

    FEATURE_NAMES = [
        "recency_days",
        "visit_frequency",
        "historical_spend",
        "cart_value",
        "competitor_browse_index",
        "churn_risk_score",
        "tenure_months",
        "is_mobile"
    ]

    TREATMENTS = [0, 10, 20]

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate(self, n_samples: int = 6000) -> pd.DataFrame:
        """
        Generates an observational dataset of customer promotional interactions.
        Returns a DataFrame containing customer attributes, observational treatment,
        observed outcome, and true counterfactual potential outcomes.
        """
        rng = self.rng

        # 1. Generate Customer Confounders X
        recency_days = rng.integers(1, 90, size=n_samples)
        visit_frequency = rng.poisson(lam=8.0, size=n_samples).clip(1, 35)
        historical_spend = np.round(rng.exponential(scale=180.0, size=n_samples) + 25.0, 2).clip(25.0, 1500.0)
        cart_value = np.round(rng.normal(loc=95.0, scale=35.0, size=n_samples).clip(20.0, 350.0), 2)
        competitor_browse_index = np.round(rng.beta(a=2.0, b=2.5, size=n_samples), 3)
        tenure_months = rng.integers(1, 48, size=n_samples)
        is_mobile = rng.binomial(n=1, p=0.62, size=n_samples)

        # Churn risk score as perceived by traditional RFM predictive model
        # Combines recency, lack of frequency, and competitor browsing
        raw_churn_risk = (
            0.45 * (recency_days / 90.0)
            - 0.35 * (visit_frequency / 35.0)
            + 0.30 * competitor_browse_index
            - 0.15 * (historical_spend / 1000.0)
            + rng.normal(0, 0.08, size=n_samples)
        )
        churn_risk_score = np.clip(1.0 / (1.0 + np.exp(-3.0 * raw_churn_risk)), 0.02, 0.98)

        X_df = pd.DataFrame({
            "recency_days": recency_days,
            "visit_frequency": visit_frequency,
            "historical_spend": historical_spend,
            "cart_value": cart_value,
            "competitor_browse_index": competitor_browse_index,
            "churn_risk_score": churn_risk_score,
            "tenure_months": tenure_months,
            "is_mobile": is_mobile
        })

        # 2. Confounded Treatment Assignment Mechanism
        # Observational policy: Marketing teams historically assigned higher discounts
        # to users who had high predicted churn risk and high competitor browsing!
        z0 = np.zeros(n_samples)
        z10 = (
            -0.6
            + 2.2 * (churn_risk_score - 0.5)
            + 1.4 * (competitor_browse_index - 0.5)
            - 0.5 * (historical_spend / 500.0)
        )
        z20 = (
            -1.2
            + 3.5 * (churn_risk_score - 0.5)
            + 2.0 * (competitor_browse_index - 0.5)
            - 0.9 * (historical_spend / 500.0)
        )

        exp_z = np.column_stack([np.exp(z0), np.exp(z10), np.exp(z20)])
        propensities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Sample observational treatment T in {0, 10, 20}
        treatment_indices = np.array([
            rng.choice([0, 1, 2], p=propensities[i])
            for i in range(n_samples)
        ])
        treatments = np.array([self.TREATMENTS[idx] for idx in treatment_indices])

        # 3. Ground Truth Archetypes and Counterfactual Potential Outcomes
        # Baseline conversion probability Y(0)
        base_logit = (
            0.5
            - 2.8 * (churn_risk_score - 0.45)
            + 0.8 * (visit_frequency / 20.0)
            - 0.6 * (recency_days / 60.0)
            + 0.5 * (historical_spend / 400.0)
            - 0.4 * (competitor_browse_index - 0.4)
        )
        p_base = 1.0 / (1.0 + np.exp(-base_logit))
        p_base = np.clip(p_base, 0.03, 0.96)

        # Identify Ground Truth Causal Archetypes
        # - Persuadables: Moderate churn risk (0.25 - 0.75), price sensitive (competitor > 0.40)
        # - Sure Things: Low churn risk (< 0.25) or high loyalty (frequency > 15, spend > 350)
        # - Lost Causes: Severe churn risk (> 0.75), high recency (> 65)
        # - Sleeping Dogs: Moderate loyalty, low price sensitivity, irritated by push promotions
        archetype = np.full(n_samples, "Persuadables", dtype=object)

        is_sure_thing = (churn_risk_score < 0.25) | ((p_base > 0.75) & (competitor_browse_index < 0.35))
        is_lost_cause = (churn_risk_score > 0.75) & (recency_days > 55) & (visit_frequency < 6)
        is_sleeping_dog = (tenure_months > 24) & (competitor_browse_index < 0.20) & (churn_risk_score > 0.35) & (churn_risk_score < 0.65)

        archetype[is_sure_thing] = "Sure Things"
        archetype[is_lost_cause] = "Lost Causes"
        archetype[is_sleeping_dog] = "Sleeping Dogs"

        # Calculate True CATE Lift for $10 and $20 discounts
        tau_10 = np.zeros(n_samples)
        tau_20 = np.zeros(n_samples)

        for i in range(n_samples):
            arch = archetype[i]
            comp = competitor_browse_index[i]
            churn = churn_risk_score[i]

            if arch == "Persuadables":
                # High positive lift that scales with price sensitivity
                t10 = 0.18 + 0.16 * comp + 0.10 * (1.0 - abs(churn - 0.5))
                t20 = 0.32 + 0.26 * comp + 0.15 * (1.0 - abs(churn - 0.5))
            elif arch == "Sure Things":
                # Tiny incremental lift, they buy anyway
                t10 = 0.02 + 0.01 * comp
                t20 = 0.03 + 0.02 * comp
            elif arch == "Lost Causes":
                # Minimal lift, they churn anyway
                t10 = 0.01 + 0.01 * comp
                t20 = 0.03 + 0.02 * comp
            elif arch == "Sleeping Dogs":
                # Negative lift (discount emails trigger cancellation/spam reaction)
                t10 = -0.07 - 0.05 * (1.0 - comp)
                t20 = -0.16 - 0.08 * (1.0 - comp)
            else:
                t10 = 0.10
                t20 = 0.18

            tau_10[i] = t10
            tau_20[i] = t20

        # Counterfactual potential outcomes Y(0), Y(10), Y(20)
        p_y0 = p_base
        p_y10 = np.clip(p_base + tau_10, 0.01, 0.99)
        p_y20 = np.clip(p_base + tau_20, 0.01, 0.99)

        # Realized probability given observed treatment
        p_realized = np.where(treatments == 0, p_y0, np.where(treatments == 10, p_y10, p_y20))
        # Binary observed conversion outcome Y in {0, 1}
        conversion = rng.binomial(n=1, p=p_realized)

        # Assemble Master DataFrame
        df = X_df.copy()
        df["customer_id"] = [f"CUST-{10001 + i}" for i in range(n_samples)]
        df["discount_offered"] = treatments
        df["converted"] = conversion
        df["true_y0_prob"] = np.round(p_y0, 4)
        df["true_y10_prob"] = np.round(p_y10, 4)
        df["true_y20_prob"] = np.round(p_y20, 4)
        df["true_tau_10"] = np.round(tau_10, 4)
        df["true_tau_20"] = np.round(tau_20, 4)
        df["archetype"] = archetype
        df["propensity_d0"] = np.round(propensities[:, 0], 4)
        df["propensity_d10"] = np.round(propensities[:, 1], 4)
        df["propensity_d20"] = np.round(propensities[:, 2], 4)

        return df


def generate_observational_pricing_dataset(n_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    """Convenience helper to generate observational pricing data."""
    generator = CustomerDataGenerator(random_seed=random_seed)
    return generator.generate(n_samples=n_samples)


if __name__ == "__main__":
    df = generate_observational_pricing_dataset(1000)
    print("Dataset generated successfully. Shape:", df.shape)
    print("Archetype distribution:\n", df["archetype"].value_counts(normalize=True))
    print("Treatment distribution:\n", df["discount_offered"].value_counts())
    print("Mean Conversion by Treatment:\n", df.groupby("discount_offered")["converted"].mean())
