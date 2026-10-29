# Dynamic Pricing via Double Machine Learning

## About the Project

This project focuses on validating the selected dataset for a dynamic pricing analysis using Double Machine Learning (DML). The goal is to assess whether observed product prices affect sales outcomes while accounting for store, product, and time-related factors.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Argparse

## Dataset

- File: wcer (1).csv
- Rows: 4,751,202
- Columns: STORE, UPC, WEEK, MOVE, PRICE

### Column Description
- STORE: Store identifier
- UPC: Product identifier
- WEEK: Time-period identifier
- PRICE: Product price
- MOVE: Sales movement

## Objective

To prepare and validate the selected dataset for causal analysis in a dynamic pricing project, ensuring the data quality is sufficient for estimating price effects using Double Machine Learning.

## Data Validation Tasks Completed

- Inspected dataset schema
- Checked for missing values
- Checked for duplicate rows
- Reviewed price and sales distributions
- Documented data limitations and project risks
- Assessed whether the available variables support causal inference

## Data Quality Findings

- No missing values were detected in the dataset
- No exact duplicate rows were detected
- Data appears structurally valid for initial analysis
- However, the dataset does not explicitly include promotion, discount, or other confounding variables, which may affect causal interpretation

## Limitations

The available variables may not fully capture external factors influencing sales, such as:
- promotions
- discounts
- competitor pricing
- stock availability
- demand shocks

These factors should be considered before drawing causal conclusions from the estimated price effect.

## Current Status

Initial data preparation and validation have been completed. Causal validation and model specification remain in progress.

## How to Run

1. Install Python.
2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

## Project Weekly Progre
- **Dataset Validation:** Thoroughly validated the dataset to ensure data quality and integrity.
- **Data Preprocessing:** Performed comprehensive data cleaning and structured the data for modeling.
- **DAG Diagram:** Designed and finalized the DAG diagram for the workflow pipeline.
-
-🚀 Week 3 Progress: Data Validation, Preprocessing & Causal Analysis(complete work process)

This week focused on validating the selected dataset, documenting data quality checks, and preparing supporting documentation for the Dynamic Pricing via Double Machine Learning (DML) project.

📋 Key Updates & Contributions

- Dataset Validation:
  
  - Reviewed the dataset schema and documented data quality checks.
  - Recorded findings related to missing values, duplicate records, and data consistency based on validation results.

- Data Preprocessing:
  
  - Added the data preprocessing notebook to support the project's data preparation workflow.
  - Documented the preprocessing work completed so far.

- Causal Analysis Documentation:
  
  - Created and uploaded a DAG diagram illustrating the assumed causal relationships.
  - Documented the proposed treatment, outcome, candidate control variables, and potential confounding risks.
  - Identified causal assumptions that require confirmation with the team.

🛠️ Commit & Branch Workflow

- Maintained project updates in the assigned working branch.
- Committed relevant documentation and project artifacts as work progressed.
