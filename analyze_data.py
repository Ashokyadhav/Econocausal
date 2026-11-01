# ============================================================
# ECONOCAUSAL: EXPLORATORY DATA ANALYSIS (EDA)
# Project: Dynamic Pricing via Double Machine Learning
# Dataset: store_scanner_pricing.csv
# Developed by: Meghana
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "store_scanner_pricing.csv"

OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

# Convert numerical columns safely
for col in ["PRICE", "MOVE"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n" + "=" * 60)
print("          ECONOCAUSAL - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\n1. DATASET OVERVIEW")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# Check required columns
required_columns = ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

print("\n2. MISSING VALUE ANALYSIS")

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_report)

missing_report.to_csv(
    f"{OUTPUT_DIR}/missing_value_report.csv"
)

print("\nRows containing missing values:")

print(df[df.isnull().any(axis=1)])


# ============================================================
# 3. DUPLICATE ANALYSIS
# ============================================================

print("\n3. DUPLICATE ANALYSIS")

duplicate_rows = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_rows)

duplicate_keys = df.duplicated(
    subset=["STORE", "UPC", "WEEK"]
).sum()

print(
    "Duplicate STORE-UPC-WEEK Combinations:",
    duplicate_keys
)


# ============================================================
# 4. STORE AND PRODUCT ANALYSIS
# ============================================================

print("\n4. STORE AND PRODUCT ANALYSIS")

print("Unique Stores:", df["STORE"].nunique())

print("Unique Products:", df["UPC"].nunique())

print("Week Range:",
      df["WEEK"].min(),
      "to",
      df["WEEK"].max())

print("\nStore Frequency:")

print(df["STORE"].value_counts())

print("\nProduct Frequency:")

print(df["UPC"].value_counts())


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\n5. STATISTICAL SUMMARY")

print("\nPRICE SUMMARY")

print(df["PRICE"].describe())

print("\nMOVE SUMMARY")

print(df["MOVE"].describe())

statistical_summary = df[
    ["PRICE", "MOVE"]
].describe()

statistical_summary.to_csv(
    f"{OUTPUT_DIR}/statistical_summary.csv"
)


# ============================================================
# 6. INVALID VALUE ANALYSIS
# ============================================================

print("\n6. INVALID VALUE ANALYSIS")

print("Missing Prices:",
      df["PRICE"].isna().sum())

print("Missing Sales:",
      df["MOVE"].isna().sum())

print("Non-positive Prices:",
      (df["PRICE"] <= 0).sum())

print("Non-positive Sales:",
      (df["MOVE"] <= 0).sum())


# ============================================================
# 7. VALID DATA FOR LOG ANALYSIS
# ============================================================

print("\n7. LOG TRANSFORMATION ANALYSIS")

valid = df.dropna(
    subset=["PRICE", "MOVE"]
).copy()

valid = valid[
    (valid["PRICE"] > 0) &
    (valid["MOVE"] > 0)
].copy()

valid["log_price"] = np.log(valid["PRICE"])

valid["log_move"] = np.log(valid["MOVE"])

print("Original Observations:", len(df))

print("Valid Observations:", len(valid))

print("Excluded Observations:", len(df) - len(valid))


# ============================================================
# 8. PRICE AND DEMAND CORRELATION
# ============================================================

print("\n8. PRICE AND DEMAND RELATIONSHIP")

if len(valid) >= 2:

    correlation = valid[
        ["PRICE", "MOVE", "log_price", "log_move"]
    ].corr()

    print("\nCorrelation Matrix:")

    print(correlation)

    log_correlation = valid[
        "log_price"
    ].corr(valid["log_move"])

    print(
        "\nLog Price vs Log Demand Correlation:",
        round(log_correlation, 4)
    )

    correlation.to_csv(
        f"{OUTPUT_DIR}/correlation_matrix.csv"
    )

else:

    print("Insufficient valid observations for correlation.")


# ============================================================
# 9. PRICE DISTRIBUTION
# ============================================================

print("\n9. PRICE DISTRIBUTION")

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="PRICE",
    bins=30,
    kde=True
)

plt.title("Distribution of Product Prices")

plt.xlabel("Product Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/price_distribution.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 10. DEMAND DISTRIBUTION
# ============================================================

print("\n10. DEMAND DISTRIBUTION")

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="MOVE",
    bins=30,
    kde=True
)

plt.title("Distribution of Product Demand")

plt.xlabel("Units Sold (MOVE)")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/demand_distribution.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 11. PRICE VS DEMAND ANALYSIS
# ============================================================

print("\n11. PRICE VS DEMAND ANALYSIS")

if not valid.empty:

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=valid,
        x="log_price",
        y="log_move",
        alpha=0.6
    )

    plt.title("Log Price vs Log Product Demand")

    plt.xlabel("Log Price")

    plt.ylabel("Log Demand")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/price_vs_demand.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# 12. STORE-WISE ANALYSIS
# ============================================================

print("\n12. STORE-WISE ANALYSIS")

store_analysis = df.groupby("STORE").agg(
    average_price=("PRICE", "mean"),
    average_demand=("MOVE", "mean"),
    total_demand=("MOVE", "sum"),
    price_std=("PRICE", "std"),
    demand_std=("MOVE", "std")
)

print(store_analysis)

store_analysis.to_csv(
    f"{OUTPUT_DIR}/store_analysis.csv"
)


# ============================================================
# 13. PRODUCT-WISE ANALYSIS
# ============================================================

print("\n13. PRODUCT-WISE ANALYSIS")

product_analysis = df.groupby("UPC").agg(
    average_price=("PRICE", "mean"),
    average_demand=("MOVE", "mean"),
    total_demand=("MOVE", "sum"),
    observations=("MOVE", "count")
)

print(product_analysis.head(20))

product_analysis.to_csv(
    f"{OUTPUT_DIR}/product_analysis.csv"
)


# ============================================================
# 14. WEEKLY PRICE AND DEMAND ANALYSIS
# ============================================================

print("\n14. WEEKLY ANALYSIS")

weekly_analysis = df.groupby("WEEK").agg(
    average_price=("PRICE", "mean"),
    average_demand=("MOVE", "mean"),
    total_demand=("MOVE", "sum")
)

print(weekly_analysis)

weekly_analysis.to_csv(
    f"{OUTPUT_DIR}/weekly_analysis.csv"
)

# Weekly trend visualization

weekly_plot = weekly_analysis.reset_index()

plt.figure(figsize=(12, 5))

plt.plot(
    range(len(weekly_plot)),
    weekly_plot["average_price"],
    marker="o",
    label="Average Price"
)

plt.title("Weekly Average Price Trend")

plt.xlabel("Week (ordered by dataset)")

plt.ylabel("Average Price")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/weekly_price_trend.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 15. OUTLIER INVESTIGATION
# ============================================================

print("\n15. OUTLIER INVESTIGATION")

print("\nTop 10 Highest Prices:")

print(
    df.nlargest(10, "PRICE")[
        ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]
    ]
)

print("\nTop 10 Highest Demand:")

print(
    df.nlargest(10, "MOVE")[
        ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]
    ]
)


# IQR-based outlier counts

for col in ["PRICE", "MOVE"]:

    series = df[col].dropna()

    if series.empty:
        print(f"\nNo valid observations for {col}.")
        continue

    Q1 = series.quantile(0.25)

    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    print(f"\n{col} Outliers:", len(outliers))

    print("Lower Bound:", lower_bound)

    print("Upper Bound:", upper_bound)


# ============================================================
# 16. FINAL EDA REPORT
# ============================================================

print("\n" + "=" * 60)

print("             FINAL EDA SUMMARY")

print("=" * 60)

print("Total Observations:", len(df))

print("Total Features:", len(df.columns))

print("Unique Stores:", df["STORE"].nunique())

print("Unique Products:", df["UPC"].nunique())

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Rows:", df.duplicated().sum())

print("Valid Log Analysis Observations:", len(valid))

print("\nGenerated Reports:")

print("1. Missing Value Report")

print("2. Statistical Summary")

print("3. Correlation Matrix")

print("4. Store Analysis")

print("5. Product Analysis")

print("6. Weekly Analysis")

print("\nGenerated Visualizations:")

print("1. Price Distribution")

print("2. Demand Distribution")

print("3. Price vs Demand")

print("4. Weekly Price Trend")

print("\nAll EDA results saved in:", OUTPUT_DIR)

print("\n========== EDA COMPLETED SUCCESSFULLY ==========")
