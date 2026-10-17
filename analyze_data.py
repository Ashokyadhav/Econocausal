import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("store_scanner_pricing.csv")

print("=== DATASET OVERVIEW ===")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 2. Missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n=== DUPLICATES ===")
print("Duplicate rows:", df.duplicated().sum())

# 4. Store and product information
print("\n=== STORES AND PRODUCTS ===")
print("Unique stores:", df["STORE"].nunique())
print("Unique UPCs:", df["UPC"].nunique())
print("Week range:", df["WEEK"].min(), "to", df["WEEK"].max())

# 5. Numerical summaries
print("\n=== PRICE SUMMARY ===")
print(df["PRICE"].describe())

print("\n=== UNITS SOLD SUMMARY ===")
print(df["MOVE"].describe())

# 6. Validate price and sales before taking logarithms
print("\n=== INVALID VALUES ===")
print("Non-positive prices:", (df["PRICE"] <= 0).sum())
print("Non-positive sales:", (df["MOVE"] <= 0).sum())

# 7. Create a valid subset for log analysis
valid = df[
    (df["PRICE"] > 0) &
    (df["MOVE"] > 0)
].copy()

valid["log_price"] = np.log(valid["PRICE"])
valid["log_move"] = np.log(valid["MOVE"])

corr = valid["log_price"].corr(valid["log_move"])

print("\n=== PRICE AND SALES RELATIONSHIP ===")
print("Valid observations:", len(valid))
print("Log-price/log-sales correlation:", round(corr, 4))

# 8. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["PRICE"].dropna(), bins=30)
plt.title("Distribution of Product Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 9. Sales distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["MOVE"].dropna(), bins=30)
plt.title("Distribution of Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 10. Price vs sales relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=valid,
    x="log_price",
    y="log_move",
    alpha=0.5
)
plt.title("Log Price vs Log Units Sold")
plt.xlabel("Log Price")
plt.ylabel("Log Units Sold")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")

# 11. Investigate missing values
print("\n=== ROWS WITH MISSING VALUES ===")
print(df[df.isnull().any(axis=1)])

# 12. Check unique products
print("\n=== UNIQUE PRODUCTS ===")
print(df["UPC"].value_counts(dropna=False))

# 13. Inspect unusually high prices
print("\n=== TOP 10 HIGHEST PRICES ===")
print(
    df.nlargest(10, "PRICE")[
        ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]
    ]
)

# 14. Inspect unusually high sales
print("\n=== TOP 10 HIGHEST SALES ===")
print(
    df.nlargest(10, "MOVE")[
        ["STORE", "UPC", "WEEK", "MOVE", "PRICE"]
    ]
)

# 15. Check store-product-week combinations
print("\n=== DUPLICATE STORE-PRODUCT-WEEK KEYS ===")
print(
    df.duplicated(
        subset=["STORE", "UPC", "WEEK"]
    ).sum()
)
