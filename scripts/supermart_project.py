# ---------------------------------------------------------
# SUPER MART GROCERY SALES ANALYSIS & PREDICTION PROJECT
# Submitted by: Ms. Bijinapally Akhila (B.Tech – Data Science)
# ---------------------------------------------------------

# -------- IMPORT REQUIRED LIBRARIES --------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------
# -------- LOAD DATASET --------
# -----------------------------------------------------
print("📌 Loading Dataset...")
data = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset (1).csv")

# Clean column names
data.columns = data.columns.str.strip()

print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# -----------------------------------------------------
# -------- DATA CLEANING --------
# -----------------------------------------------------
print("\n🧹 Starting Data Cleaning...")

# Convert dates
data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")

# Drop rows without dates
data.dropna(subset=["Order Date"], inplace=True)

# Extract date features
data["Order Day"] = data["Order Date"].dt.day
data["Order Month"] = data["Order Date"].dt.month
data["Order Year"] = data["Order Date"].dt.year

# Fill missing values
data.fillna(0, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

print("✅ Data Cleaning Completed!")
print("\nColumns:", data.columns.tolist())

# -----------------------------------------------------
# -------- EXPLORATORY DATA ANALYSIS (EDA) --------
# -----------------------------------------------------
print("\n📊 Starting EDA...")

# ==========================
# 1️⃣ Sales by Category
# ==========================
plt.figure(figsize=(8,5))
sns.barplot(x="Category", y="Sales", data=data, estimator=sum, palette="Set2")
plt.title("Sales Distribution by Category", fontsize=14, weight='bold')
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.tight_layout()
plt.savefig("sales_by_category.png")
plt.close()

# ==========================
# 2️⃣ Monthly Sales Trend
# ==========================
plt.figure(figsize=(10,5))
monthly_sales = data.groupby("Order Month")["Sales"].sum()
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend", fontsize=14, weight='bold')
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_trend.png")
plt.close()

# ==========================
# 3️⃣ Correlation Heatmap
# ==========================
plt.figure(figsize=(8,5))
sns.heatmap(data.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# ==========================
# 4️⃣ Top 10 Selling Sub-Categories
# ==========================
plt.figure(figsize=(10,5))
top_products = data.groupby("Sub Category")["Sales"].sum().sort_values(ascending=False)[:10]
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Most Selling Sub-Categories", fontsize=14, weight='bold')
plt.xlabel("Total Sales")
plt.ylabel("Sub Category")
plt.tight_layout()
plt.savefig("top10_subcategories.png")
plt.close()

# ==========================
# 5️⃣ Sales by Region
# ==========================
plt.figure(figsize=(8,5))
region_sales = data.groupby("Region")["Sales"].sum().reset_index()
sns.barplot(x="Region", y="Sales", data=region_sales, palette="Accent")
plt.title("Sales by Region", fontsize=14, weight='bold')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("sales_by_region.png")
plt.close()

# ==========================
# 6️⃣ Profit vs Sales Scatter Plot
# ==========================
plt.figure(figsize=(7,6))
sns.scatterplot(x="Sales", y="Profit", data=data, hue="Category", palette="Set2")
plt.title("Profit vs Sales", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("profit_vs_sales.png")
plt.close()

# ==========================
# 7️⃣ Category-wise Profit
# ==========================
plt.figure(figsize=(8,5))
sns.barplot(x="Category", y="Profit", data=data, palette="coolwarm")
plt.title("Profit by Category", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("profit_by_category.png")
plt.close()

# ==========================
# 8️⃣ Monthly Profit Trend
# ==========================
plt.figure(figsize=(10,5))
monthly_profit = data.groupby("Order Month")["Profit"].sum()
monthly_profit.plot(kind="line", marker="o", color="purple")
plt.title("Monthly Profit Trend", fontsize=14, weight='bold')
plt.xlabel("Month")
plt.ylabel("Total Profit")
plt.tight_layout()
plt.savefig("monthly_profit_trend.png")
plt.close()

# ==========================
# 9️⃣ Discount vs Sales
# ==========================
plt.figure(figsize=(7,6))
sns.scatterplot(x="Discount", y="Sales", data=data, color="steelblue")
plt.title("Effect of Discount on Sales", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("discount_vs_sales.png")
plt.close()

print("\n✅ EDA Completed – Graphs Saved:")
print("   • sales_by_category.png")
print("   • sales_trend.png")
print("   • correlation_heatmap.png")
print("   • top10_subcategories.png")
print("   • sales_by_region.png")
print("   • profit_vs_sales.png")
print("   • profit_by_category.png")
print("   • monthly_profit_trend.png")
print("   • discount_vs_sales.png")

# -----------------------------------------------------
# -------- MACHINE LEARNING MODELS --------
# -----------------------------------------------------
print("\n🤖 Starting Machine Learning Models...")

# Select target and features
X = data.drop(columns=["Order ID", "Customer Name", "Order Date", "Sales"])
y = data["Sales"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Fix missing values
X = X.fillna(0)
y = y.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------- Linear Regression --------
print("\n📌 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Linear Regression MSE:", round(mean_squared_error(y_test, lr_pred), 2))
print("Linear Regression R² Score:", round(r2_score(y_test, lr_pred), 2))

# -------- Random Forest --------
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest MSE:", round(mean_squared_error(y_test, rf_pred), 2))
print("Random Forest R² Score:", round(r2_score(y_test, rf_pred), 2))

# -------- Save Accuracy Graph --------
plt.figure(figsize=(7,6))
plt.scatter(y_test, rf_pred, alpha=0.5, color="green")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Random Forest: Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("rf_actual_vs_pred.png")
plt.close()

print("\n✅ Machine Learning Completed!")
print("   • rf_actual_vs_pred.png Generated")

print("\n📌 Project Completed Successfully by: Ms. Bijinapally Akhila")