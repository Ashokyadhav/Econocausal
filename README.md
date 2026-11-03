# EconoCausal – Dynamic Pricing Using Double Machine Learning
# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
# Econocausal
Dynamic Pricing via Double Machine Learning
# Data Preparation

**An AI/ML-based project for analysing coupon effectiveness, estimating treatment effects, and optimizing marketing budget allocation.**

## 1. Project Overview

EconoCausal is a data science and machine learning project that studies how coupon offers influence customer purchasing behaviour.

The project combines Exploratory Data Analysis (EDA), Double Machine Learning (DML), and Uplift Modelling to understand coupon effectiveness and identify customers who may respond positively to promotional offers.

The main goal is to help businesses make better coupon distribution decisions and allocate marketing budgets more efficiently.

## 2. Problem Statement

Businesses frequently provide coupons and discounts to encourage customers to make purchases.

However, giving coupons to every customer may increase marketing expenses without generating additional purchases.

Therefore, businesses need a way to:

- Understand the effect of coupons on customer purchases.
- Estimate the average treatment effect of coupon offers.
- Identify customers who may respond positively to coupons.
- Compare different coupon strategies.
- Improve marketing budget allocation.

EconoCausal addresses these challenges using causal machine learning and uplift modelling techniques.

## 3. Project Objectives

- Analyse customer purchasing behaviour.
- Clean and prepare the customer dataset.
- Compare treatment and control groups.
- Estimate coupon effects using Double Machine Learning.
- Identify customer-level uplift using T-Learner.
- Segment customers based on predicted uplift.
- Develop a budget allocation framework.
- Provide data-driven business recommendations.

## 4. Dataset Description

**Dataset:** `Coupon_Impact_Clean.csv`

**Dataset size:** 12,959 rows × 8 columns.

The dataset contains customer-level information and coupon experiment groups.

### Dataset columns

| Column | Description |
|---|---|
| consumer_id | Unique customer identifier |
| group | Experimental group |
| amount | Purchase amount |
| purchase | Purchase indicator |
| redeem | Coupon redemption indicator |
| pageviews | Number of page views |
| visit | Customer visit indicator |
| discount | Discount information |

### Experimental Groups

| Group | Description |
|---|---|
| CGL | Low-tier control group |
| T1L | Low-tier treatment group |
| CGH | High-tier control group |
| T1H | High-tier treatment group |
| T2H | Additional high-tier treatment group |

## 5. Project Architecture

The project consists of three major modules.

### Module 1: Data Cleaning and Exploratory Data Analysis

The first module focuses on understanding and preparing the dataset.

Activities performed:

- Dataset loading using Pandas.
- Data structure inspection.
- Missing value checking.
- Duplicate record checking.
- Data type inspection.
- Treatment and control group analysis.
- Purchase rate comparison.
- Average purchase amount analysis.
- Exploratory data visualizations.

**Outcome:** A cleaned and analysed dataset ready for further modelling.

### Module 2: Double Machine Learning (DML)

Double Machine Learning is used to estimate the effect of coupon treatments on customer purchasing behaviour while accounting for relevant observed factors.

**Methodology:**

- DoubleML library.
- Partial Linear Regression (PLR).
- Random Forest Regression for the outcome model.
- Logistic Regression for the treatment model.
- Five-fold cross-validation.

**Treatment:** Coupon treatment.

**Main outcomes:**
- Purchase.
- Purchase amount.

Four comparisons were performed:

1. High-tier treatment versus high-tier control – Purchase amount.
2. High-tier treatment versus high-tier control – Purchase.
3. Low-tier treatment versus low-tier control – Purchase amount.
4. Low-tier treatment versus low-tier control – Purchase.

**Evaluation metrics:**
- Average Treatment Effect (ATE).
- Standard Error.
- P-value.
- Confidence Interval.
- Sensitivity Analysis.

### Module 3: Uplift Modelling and Budget Optimization

This module identifies customers who may respond positively to coupon offers.

**Uplift Modelling:**

We implemented the T-Learner approach.

T-Learner uses two separate models:

- Control Model: Predicts purchase probability without treatment.
- Treatment Model: Predicts purchase probability with treatment.

The difference between the two predictions gives the estimated uplift score.

Three comparisons were developed:

1. Low-tier: T1L versus CGL.
2. High-tier T1: T1H versus CGH.
3. High-tier T2: T2H versus CGH.

**Customer Segmentation:**

Customers were divided into three segments:

- Negative Uplift.
- Low/Moderate Uplift.
- High Uplift.

**Budget Optimization:**

The budget optimization module:

- Ranks customers using predicted uplift scores.
- Prioritizes customers with positive predicted uplift.
- Selects customers within an illustrative budget limit.
- Estimates incremental purchases.
- Compares budget efficiency across coupon strategies.

## 6. Key Results

### Double Machine Learning Results

The following are the recorded DML estimates from our analysis.

| Comparison | ATE | P-value |
|---|---:|---:|
| High-tier – Amount | -0.2110 | 0.5000 |
| High-tier – Purchase | -0.0084 | 0.3374 |
| Low-tier – Amount | -0.0778 | 0.3187 |
| Low-tier – Purchase | -0.0036 | 0.2279 |

**Interpretation:**

All four p-values are greater than 0.05. Therefore, our analysis did not find statistically significant evidence of a coupon treatment effect.

This does not prove that coupons have no effect.

### Uplift Modelling Results

Average predicted uplift:

| Strategy | Average Predicted Uplift |
|---|---:|
| Low-tier | 0.01018 |
| High-tier T1 | -0.00574 |
| High-tier T2 | 0.01162 |

High-tier T2 showed the highest average predicted uplift among the three comparisons.

### Budget Optimization Results

The following results use illustrative budget and coupon-cost assumptions.

| Strategy | Customers Targeted | Expected Incremental Purchases |
|---|---:|---:|
| Low-tier | 1,000 | 20.52 |
| High-tier T1 | 156 | 2.89 |
| High-tier T2 | 433 | 17.79 |

**Key observations:**

- High-tier T2 showed the highest estimated purchases per budget unit.
- Low-tier showed the highest estimated incremental purchase volume.
- Uplift-based targeting can help prioritize customers for coupon offers.

These are model-based estimates, not confirmed business profits, savings, or causal business outcomes.

## 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| DoubleML | Causal machine learning |
| Logistic Regression | Treatment and uplift modelling |
| Random Forest | DML outcome modelling |
| Streamlit | Interactive application/dashboard |
| Git and GitHub | Version control and collaboration |

## 8. Project Workflow

```text
Coupon Impact Dataset
        |
        v
Data Cleaning and EDA
        |
        v
Treatment and Control Analysis
        |
        v
Double Machine Learning
        |
        v
Average Treatment Effect
        |
        v
Uplift Modelling (T-Learner)
        |
        v
Customer Segmentation
        |
        v
Budget Optimization
        |
        v
Business Recommendations
        |
        v
Dashboard and Final Results
```

## 9. Project Structure

Suggested repository organization:

```text
Econocausal/
│
├── data/
│   └── Coupon_Impact_Clean.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── DoubleML.ipynb
│   └── Budget_Optimization.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

The actual repository structure may differ depending on the team's final integration.

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/Ashokyadhav/Econocausal.git
```

### Step 2: Open the project

```bash
cd Econocausal
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn doubleml streamlit
```

### Step 4: Open the notebooks

Open the project notebooks using Google Colab or Jupyter Notebook.

Upload or connect the dataset as required.

### Step 5: Run the analysis

Execute the notebooks in the following order:

1. Data Cleaning and EDA.
2. Double Machine Learning.
3. Uplift Modelling and Budget Optimization.

### Step 6: Run the dashboard

If the integrated Streamlit application is available:

```bash
streamlit run app.py
```

## 11. Business Recommendations

Based on the project analysis:

- Use uplift modelling to identify customers who may respond positively to coupons.
- Avoid distributing coupons without considering expected incremental benefit.
- Compare different coupon strategies before allocating marketing budgets.
- Validate predicted treatment effects through controlled experiments.
- Use actual coupon costs and business budgets for practical optimization.

## 12. Limitations

- Treatment assignment and experimental design assumptions require verification.
- The timing of predictor variables must be confirmed to avoid post-treatment leakage.
- Predicted uplift is not automatically a verified causal effect.
- Coupon costs and budget limits used in the optimization are illustrative assumptions.
- Further validation is required before real-world deployment.

## 13. Future Enhancements

- Validate treatment assignment and experiment design.
- Improve uplift model performance using additional suitable algorithms.
- Evaluate models using validated uplift and Qini metrics.
- Integrate actual coupon costs and business budget constraints.
- Develop an interactive Streamlit dashboard.
- Test the recommendations using controlled experiments.

## 14. Conclusion

EconoCausal combines data analysis, Double Machine Learning, and Uplift Modelling to study coupon effectiveness and customer purchasing behaviour.

The project estimates average treatment effects, identifies customers with potentially positive treatment responses, and compares coupon strategies under illustrative budget assumptions.

The overall objective is to support better marketing decisions through data-driven coupon targeting and budget allocation.

## 15. Contributors

**Project:** EconoCausal – Dynamic Pricing Using Double Machine Learning

**GitHub Repository:**  
https://github.com/Ashokyadhav/Econocausal

**Team:** 4 Members

---

*This project was developed for academic and practical learning purposes as part of a collaborative data science project.*
