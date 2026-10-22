"""
EconoCausal - Dynamic Pricing & Campaign Budget Optimizer
Solves the profit-maximizing discount allocation policy under budget constraints.
Compares Causal DML Policy vs Traditional Predictive Churn Policy vs Blanket Promotion.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class DynamicPricingOptimizer:
    """
    Solves optimal discount allocation under marketing budget constraints.
    Calculates incremental profit:
    Delta_Profit_i(d) = tau_{i,d} * (Price - Cost - d) - d * Y_{i,0}
    """

    def __init__(self, base_price: float = 99.0, unit_cost: float = 39.0):
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.base_margin = base_price - unit_cost  # E.g. $60 margin

    def evaluate_customer_incremental_profit(self, y0: float, tau_10: float, tau_20: float) -> Dict[str, float]:
        """
        Calculates expected incremental profit over no-discount baseline for an individual.
        """
        # Incremental profit for $10 discount:
        # tau_10 * (Margin - 10) - 10 * y0
        margin_10 = self.base_margin - 10.0
        profit_10 = tau_10 * margin_10 - 10.0 * y0

        # Incremental profit for $20 discount:
        # tau_20 * (Margin - 20) - 20 * y0
        margin_20 = self.base_margin - 20.0
        profit_20 = tau_20 * margin_20 - 20.0 * y0

        return {
            "profit_0": 0.0,
            "profit_10": float(profit_10),
            "profit_20": float(profit_20),
            "best_action": 20 if (profit_20 > profit_10 and profit_20 > 0) else (10 if profit_10 > 0 else 0)
        }

    def optimize_campaign(
        self,
        df: pd.DataFrame,
        budget: float = 40000.0,
        base_price: float = 99.0,
        unit_cost: float = 39.0
    ) -> Dict[str, Any]:
        """
        Simulates and compares three targeting strategies across the dataset:
        1. EconoCausal Causal Policy (DML)
        2. Traditional Churn Predictive ML Policy
        3. Blanket Promotion Policy
        """
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.base_margin = base_price - unit_cost
        n_samples = len(df)

        y0 = df["pred_y0_prob"].values if "pred_y0_prob" in df.columns else df["true_y0_prob"].values
        tau_10 = df["pred_tau_10"].values if "pred_tau_10" in df.columns else df["true_tau_10"].values
        tau_20 = df["pred_tau_20"].values if "pred_tau_20" in df.columns else df["true_tau_20"].values
        churn_risk = df["churn_risk_score"].values

        # 1. Compute Individual Incremental Profit for each option
        margin_10 = self.base_margin - 10.0
        margin_20 = self.base_margin - 20.0

        inc_profit_10 = tau_10 * margin_10 - 10.0 * y0
        inc_profit_20 = tau_20 * margin_20 - 20.0 * y0

        # Expected discount payout cost: (y0 + tau_d) * d
        cost_10 = np.clip(y0 + tau_10, 0.01, 1.0) * 10.0
        cost_20 = np.clip(y0 + tau_20, 0.01, 1.0) * 20.0

        # -------------------------------------------------------------
        # Policy 1: EconoCausal (Causal DML Policy)
        # -------------------------------------------------------------
        # Determine best action per customer based on ROI efficiency
        # Efficiency = Incremental Profit / Cost
        eff_10 = np.where(cost_10 > 0, inc_profit_10 / cost_10, -999.0)
        eff_20 = np.where(cost_20 > 0, inc_profit_20 / cost_20, -999.0)

        # Candidates for $20 vs $10 vs $0
        dml_assigned = np.zeros(n_samples, dtype=int)
        dml_inc_profit = np.zeros(n_samples)
        dml_spent = np.zeros(n_samples)
        dml_conversions = np.zeros(n_samples)
        dml_wasted_spend = np.zeros(n_samples)

        # Priority score: highest positive incremental profit per dollar spent
        priority_scores = np.maximum(
            np.where(inc_profit_20 > 0, inc_profit_20 / cost_20, -1.0),
            np.where(inc_profit_10 > 0, inc_profit_10 / cost_10, -1.0)
        )
        dml_order = np.argsort(priority_scores)[::-1]

        current_budget = budget
        for idx in dml_order:
            if current_budget <= 0:
                break
            p20 = inc_profit_20[idx]
            p10 = inc_profit_10[idx]
            c20 = cost_20[idx]
            c10 = cost_10[idx]

            # Choose best profitable action that fits within remaining budget
            if p20 > p10 and p20 > 0 and current_budget >= c20:
                dml_assigned[idx] = 20
                dml_inc_profit[idx] = p20
                dml_spent[idx] = c20
                dml_conversions[idx] = tau_20[idx]
                current_budget -= c20
            elif p10 > 0 and current_budget >= c10:
                dml_assigned[idx] = 10
                dml_inc_profit[idx] = p10
                dml_spent[idx] = c10
                dml_conversions[idx] = tau_10[idx]
                current_budget -= c10

        # -------------------------------------------------------------
        # Policy 2: Traditional Predictive ML Policy (Targets Top Churners)
        # -------------------------------------------------------------
        # Traditional marketing allocates highest discounts ($20) to highest churn risk users
        churn_order = np.argsort(churn_risk)[::-1]
        churn_assigned = np.zeros(n_samples, dtype=int)
        churn_inc_profit = np.zeros(n_samples)
        churn_spent = np.zeros(n_samples)
        churn_conversions = np.zeros(n_samples)
        churn_wasted_spend = np.zeros(n_samples)

        cur_b_churn = budget
        for idx in churn_order:
            if cur_b_churn <= 0:
                break
            c20 = cost_20[idx]
            if cur_b_churn >= c20:
                churn_assigned[idx] = 20
                churn_inc_profit[idx] = inc_profit_20[idx]
                churn_spent[idx] = c20
                churn_conversions[idx] = tau_20[idx]
                if inc_profit_20[idx] <= 0:
                    churn_wasted_spend[idx] = c20
                cur_b_churn -= c20
            elif cur_b_churn >= cost_10[idx]:
                churn_assigned[idx] = 10
                churn_inc_profit[idx] = inc_profit_10[idx]
                churn_spent[idx] = cost_10[idx]
                churn_conversions[idx] = tau_10[idx]
                if inc_profit_10[idx] <= 0:
                    churn_wasted_spend[idx] = cost_10[idx]
                cur_b_churn -= cost_10[idx]

        # -------------------------------------------------------------
        # Policy 3: Blanket Promotion Policy (Random / Mass Discount)
        # -------------------------------------------------------------
        blanket_order = np.random.default_rng(42).permutation(n_samples)
        blanket_assigned = np.zeros(n_samples, dtype=int)
        blanket_inc_profit = np.zeros(n_samples)
        blanket_spent = np.zeros(n_samples)
        blanket_conversions = np.zeros(n_samples)
        blanket_wasted_spend = np.zeros(n_samples)

        cur_b_blanket = budget
        for idx in blanket_order:
            if cur_b_blanket <= 0:
                break
            # Gives $10 discount across the board
            c10 = cost_10[idx]
            if cur_b_blanket >= c10:
                blanket_assigned[idx] = 10
                blanket_inc_profit[idx] = inc_profit_10[idx]
                blanket_spent[idx] = c10
                blanket_conversions[idx] = tau_10[idx]
                if inc_profit_10[idx] <= 0:
                    blanket_wasted_spend[idx] = c10
                cur_b_blanket -= c10

        # Helper to compile policy performance stats
        def build_stats(assigned, inc_profit, spent, conv, wasted):
            tot_spent = float(np.sum(spent))
            tot_profit = float(np.sum(inc_profit))
            tot_conv = float(np.sum(conv))
            tot_wasted = float(np.sum(wasted))
            gross_inc_rev = tot_profit + tot_spent
            roi = (tot_profit / tot_spent * 100.0) if tot_spent > 0 else 0.0

            n_treated = int(np.sum(assigned > 0))
            n_d10 = int(np.sum(assigned == 10))
            n_d20 = int(np.sum(assigned == 20))

            return {
                "targeted_customers": n_treated,
                "discounts_10_count": n_d10,
                "discounts_20_count": n_d20,
                "budget_spent": round(tot_spent, 2),
                "incremental_conversions": round(tot_conv, 1),
                "gross_incremental_revenue": round(gross_inc_rev, 2),
                "net_incremental_profit": round(tot_profit, 2),
                "wasted_spend": round(tot_wasted, 2),
                "roi_percentage": round(roi, 1),
                "cpa_incremental": round((tot_spent / tot_conv), 2) if tot_conv > 0 else 0.0
            }

        dml_stats = build_stats(dml_assigned, dml_inc_profit, dml_spent, dml_conversions, dml_wasted_spend)
        churn_stats = build_stats(churn_assigned, churn_inc_profit, churn_spent, churn_conversions, churn_wasted_spend)
        blanket_stats = build_stats(blanket_assigned, blanket_inc_profit, blanket_spent, blanket_conversions, blanket_wasted_spend)

        # Delta comparisons
        profit_advantage = round(dml_stats["net_incremental_profit"] - churn_stats["net_incremental_profit"], 2)
        waste_saved = round(churn_stats["wasted_spend"] - dml_stats["wasted_spend"], 2)
        conv_advantage = round(dml_stats["incremental_conversions"] - churn_stats["incremental_conversions"], 1)

        return {
            "parameters": {
                "budget": budget,
                "base_price": base_price,
                "unit_cost": unit_cost,
                "base_margin": self.base_margin
            },
            "policies": {
                "dml_causal_policy": dml_stats,
                "traditional_churn_policy": churn_stats,
                "blanket_promotion_policy": blanket_stats
            },
            "executive_delta": {
                "profit_advantage_usd": profit_advantage,
                "waste_eliminated_usd": waste_saved,
                "additional_conversions": conv_advantage,
                "roi_spread_pts": round(dml_stats["roi_percentage"] - churn_stats["roi_percentage"], 1)
            },
            "targeting_assignments": {
                "customer_ids": df["customer_id"].tolist() if "customer_id" in df.columns else [f"CUST-{i}" for i in range(n_samples)],
                "assigned_discount": dml_assigned.tolist()
            }
        }
