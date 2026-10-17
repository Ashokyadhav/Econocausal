 EconoCausal: Dynamic Pricing Using Double Machine Learning

## About the Project

EconoCausal is a machine learning project that explores how businesses can decide which discounts to offer to customers.

Normally, a machine learning model predicts whether a customer is likely to purchase a product. However, a customer might purchase the product even without a discount.

Our project focuses on understanding whether offering a discount actually changes the customer's decision to purchase.

The goal is to use causal machine learning to support better discount decisions and make marketing budgets more effective.

## Problem Statement

Businesses often provide discounts to customers without knowing whether those discounts will lead to additional purchases.

For example, a customer may already intend to purchase a product. Giving that customer a discount could reduce the business's profit without increasing sales.

Our project aims to study the effect of discounts on customer purchasing behaviour and explore how this information can support better pricing decisions.

## Objectives

- Explore and understand the available dataset.
- Identify important features related to customer purchases and discounts.
- Perform data cleaning and exploratory data analysis.
- Study the relationship between discounts and purchasing behaviour.
- Explore Double Machine Learning for estimating the effect of discounts.
- Support better discount allocation using the estimated effects.

## Technologies Used

The project uses Python for data analysis and machine learning.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading and analysing datasets |
| NumPy | Numerical operations |
| Matplotlib | Creating data visualisations |
| Scikit-learn | Machine learning utilities, where required |
| Double Machine Learning | Estimating the causal effect of discounts |

The tools listed above describe the intended analysis workflow. The actual libraries used may vary as the project develops.

## My Contribution: Exploratory Data Analysis

I am contributing to the EconoCausal project by working on the data analysis stage.

My current contribution includes the `analyze_data.py` script.

The purpose of this stage is to examine the data before developing and evaluating machine learning models.

The analysis workflow may include:

1. Loading the dataset.
2. Checking the number of rows and columns.
3. Examining column names and data types.
4. Checking for missing values and duplicate records.
5. Understanding numerical and categorical features.
6. Exploring patterns that may be useful for further analysis.

The completed steps and findings will be documented as the analysis develops.

## How Double Machine Learning Fits into the Project

Double Machine Learning is a method used to estimate causal effects while accounting for other variables that may influence the outcome.

In our project, customer characteristics and previous purchasing behaviour may influence both discount assignment and purchasing decisions.

The method can help us investigate the effect of a discount while accounting for relevant observed characteristics.

This is important because a relationship between discounts and purchases does not automatically mean that the discount caused the purchase.

## Expected Outcome

The project aims to develop a better understanding of how discounts influence customer purchasing decisions.

The analysis will support further work on causal effect estimation and data-driven discount allocation.

The final results will depend on the dataset, the implemented models, and the evaluation process.

## Project Status

The project is under development.

The EDA work is being developed and integrated with the team's existing code. Further analysis, model development, evaluation, and integration will be completed as the project progresses.

## Repository Contents

| File | Description |
|---|---|
| `Econocasual.ipynb` | Existing project notebook |
| `analyze_data.py` | Python script for data analysis |
| `README.md` | Project documentation |

Additional files may be added as the project develops.

## Team Contribution

This is a collaborative academic project. Individual contributions are tracked through Git commits and pull requests.

My current contribution is the development of the initial data analysis script.

## Conclusion

EconoCausal explores how causal machine learning can help businesses make more informed discount decisions.

By understanding the data first and then estimating the effect of discounts, the project aims to move beyond simply predicting customer purchases toward understanding how pricing decisions may influence them.
