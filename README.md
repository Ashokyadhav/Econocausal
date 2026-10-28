# Econocausal
Dynamic Pricing via Double Machine Learning
# Data Preparation

## Objective
Prepare and validate the selected dataset for the Dynamic Pricing via Double Machine Learning project to ensure data readiness and assess its suitability for causal inference.

## Responsibilities
- Document dataset specifications, source, and variables
- Validate data types, handle missing values, and remove duplicate records
- Detect anomalies in price distributions, sales volumes, and outliers
- Assess whether the available variables are sufficient for causal price-effect analysis
- Report data-quality issues, confounding gaps, and project risks

## Dataset Details

| Attribute | Details |
| :--- | :--- |
| File Name | `wcer (1).csv` |
| Total Rows | 4,751,202 |
| Total Columns | 5 |
| Purpose | Analyze store-product sales and prices over time |

### Variable Overview
- `STORE`: Unique store identifier
- `UPC`: Universal Product Code (product identifier)
- `WEEK`: Weekly time variable
- `MOVE`: Sales volume (units sold)
- `PRICE`: Unit selling price

## Current Status
The dataset has been identified and ingested into the project pipeline. Initial data-quality checks and variable analyses have been completed to establish baseline readiness for the upcoming Double Machine Learning (DML) stage.

## Data Limitations and Causal Considerations
- Unobserved confounders: The dataset does not include explicit indicators for promotional campaigns, discounts, local advertising, or competitor pricing.
- Causal risk: Because key confounding variables are missing, direct estimates may be affected by omitted variable bias. These limitations should be accounted for during model design before interpreting price effects as causal.

## Next Steps
1. Perform final verification of all data-quality checks on the processed dataset.
2. Formally document confirmed validation results and feature transformations for team reference.
