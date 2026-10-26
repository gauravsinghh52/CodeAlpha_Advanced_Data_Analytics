from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced EDA - Sales Analytics
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Create reproducible sample data if no dataset exists
data_file = BASE_DIR / "sales_data.csv"

if data_file.exists():
    df = pd.read_csv(data_file)
    print(f"Loaded dataset: {data_file.name}")
else:
    n = 300
    categories = ["Electronics", "Accessories", "Home", "Clothing"]
    cities = ["Delhi", "Mumbai", "Patna", "Bengaluru", "Kolkata"]
    products = ["Product A", "Product B", "Product C", "Product D"]

    df = pd.DataFrame({
        "OrderID": range(1001, 1001 + n),
        "Date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Product": np.random.choice(products, n),
        "Category": np.random.choice(categories, n),
        "City": np.random.choice(cities, n),
        "Quantity": np.random.randint(1, 8, n),
        "UnitPrice": np.random.uniform(200, 15000, n).round(2)
    })

    df["Sales"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df.to_csv(data_file, index=False)
    print("Created sample sales_data.csv")

# Data cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for column in ["Quantity", "UnitPrice", "Sales"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["Category", "City", "Sales"])
df = df[df["Sales"] >= 0]

if df.empty:
    raise ValueError("No valid rows remain after data cleaning.")

# Save cleaned data
df.to_csv(OUTPUT_DIR / "cleaned_sales_data.csv", index=False)

# KPIs and category/city summaries
total_sales = df["Sales"].sum()
average_sale = df["Sales"].mean()
total_orders = len(df)

category_sales = df.groupby("Category")["Sales"].sum().sort_values(
    ascending=False
)
city_sales = df.groupby("City")["Sales"].sum().sort_values(
    ascending=False
)

category_sales.to_csv(OUTPUT_DIR / "sales_by_category.csv")
city_sales.to_csv(OUTPUT_DIR / "sales_by_city.csv")

# Chart 1: Sales by category
plt.figure(figsize=(9, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(CHART_DIR / "sales_by_category.png", dpi=150)
plt.show()
plt.close()

# Chart 2: Sales by city
plt.figure(figsize=(9, 5))
sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Sales")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(CHART_DIR / "sales_by_city.png", dpi=150)
plt.show()
plt.close()

# Chart 3: Sales distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["Sales"], bins=25, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(CHART_DIR / "sales_distribution.png", dpi=150)
plt.show()
plt.close()

# Chart 4: Numeric correlation heatmap
numeric_df = df.select_dtypes(include="number")
if len(numeric_df.columns) >= 2:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()

# Generate report
best_category = category_sales.index[0]
best_city = city_sales.index[0]

report = f"""
ADVANCED SALES EDA REPORT
=========================
Rows analysed: {len(df)}
Duplicate rows removed: Check cleaned dataset against original
Missing values remaining: {int(df.isnull().sum().sum())}

TOTAL SALES: {total_sales:,.2f}
AVERAGE SALE: {average_sale:,.2f}
TOTAL ORDERS/ROWS: {total_orders}

TOP CATEGORY: {best_category}
TOP CITY: {best_city}

SALES BY CATEGORY:
{category_sales.to_string()}

SALES BY CITY:
{city_sales.to_string()}
"""

(OUTPUT_DIR / "summary_report.txt").write_text(report, encoding="utf-8")

print(report)
print("\nTask 2 completed successfully!")
print(f"Outputs saved in: {OUTPUT_DIR}")