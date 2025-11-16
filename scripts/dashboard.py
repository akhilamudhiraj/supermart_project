# dashboard.py
# Neon Electric Blue Streamlit Dashboard for Supermart Grocery Sales
# Uses file: "data/Supermart Grocery Sales - Retail Analytics Dataset (1).csv"
# Built for Ms. Bijinapally Akhila

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Supermart Grocery Sales Dashboard", layout="wide", initial_sidebar_state="collapsed")

NEON_CSS = """
<style>
html, body { font-family:'Montserrat', sans-serif; background: #0A0A0A; color:#FFDFAA; }
.stApp { background: linear-gradient(140deg, #000000, #141414, #1F1A13); }
.neon-title { text-align:center; font-size:38px; font-weight:700; color:#FFD700; text-shadow:0 0 10px #FFDD55, 0 0 20px #FFB300; }
.kpi-card { background:#151515; border:1px solid #FFB300; border-radius:12px; padding:15px; box-shadow:0 0 18px rgba(255,180,0,0.3); }
.footer { text-align:center; margin-top:20px; font-size:13px; color:#9FDFFF; }
</style>
"""

st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 class='neon-title'> Supermart Grocery Sales Dashboard </h1>", unsafe_allow_html=True)
st.write("")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Supermart Grocery Sales - Retail Analytics Dataset (1).csv")  # updated path
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Year"] = df["Order Date"].dt.year
    df.fillna(0, inplace=True)
    return df

data = load_data()

# ---------------- FILTERS ----------------
with st.container():
    cols = st.columns([1, 2, 2, 2, 2, 1])
    with cols[1]:
        cat_options = sorted(data["Category"].unique())
        category_filter = st.multiselect("Category", options=cat_options)
    with cols[2]:
        sub_options = sorted(data["Sub Category"].unique())
        sub_filter = st.multiselect("Sub Category", options=sub_options)
    with cols[3]:
        reg_options = sorted(data["Region"].unique())
        region_filter = st.multiselect("Region", options=reg_options)
    with cols[4]:
        year_options = sorted([int(x) for x in data["Order Year"].unique() if x != 0])
        year_filter = st.multiselect("Year", options=year_options)

# ---------------- APPLY FILTERS ----------------
filtered = data.copy()
if category_filter:
    filtered = filtered[filtered["Category"].isin(category_filter)]
if sub_filter:
    filtered = filtered[filtered["Sub Category"].isin(sub_filter)]
if region_filter:
    filtered = filtered[filtered["Region"].isin(region_filter)]
if year_filter:
    filtered = filtered[filtered["Order Year"].isin(year_filter)]

# ---------------- KPIs ----------------
st.write("")
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"<div class='kpi-card'><h4>Total Sales</h4><h2 style='color:#BEEBFF'>₹{filtered['Sales'].sum():,.2f}</h2></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><h4>Total Profit</h4><h2 style='color:#BEEBFF'>₹{filtered['Profit'].sum():,.2f}</h2></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><h4>Avg Discount</h4><h2 style='color:#BEEBFF'>{filtered['Discount'].mean():.2f}%</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- CHARTS ----------------
rows = [st.columns(3) for _ in range(3)]
charts = [
    ("Sales by Category", "Category", "Sales", "bar"),
    ("Monthly Sales Trend", "Order Month", "Sales", "line"),
    ("Correlation Heatmap", None, None, "heatmap"),
    ("Top 10 Sub-Categories", "Sub Category", "Sales", "hbar"),
    ("Sales by Region", "Region", "Sales", "bar"),
    ("Profit vs Sales", "Sales", "Profit", "scatter"),
    ("Profit by Category", "Category", "Profit", "bar"),
    ("Monthly Profit Trend", "Order Month", "Profit", "line"),
    ("Discount vs Sales", "Discount", "Sales", "scatter")
]

for i, (title, x_col, y_col, chart_type) in enumerate(charts):
    col = rows[i//3][i%3]
    with col:
        if chart_type == "bar":
            df = filtered.groupby(x_col)[y_col].sum().reset_index()
            fig = px.bar(df, x=x_col, y=y_col, title=title, template="plotly_dark", color=x_col)
        elif chart_type == "hbar":
            df = filtered.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df, x=y_col, y=x_col, orientation="h", title=title, template="plotly_dark")
        elif chart_type == "line":
            df = filtered.groupby(x_col)[y_col].sum().reset_index()
            fig = px.line(df, x=x_col, y=y_col, markers=True, title=title, template="plotly_dark")
        elif chart_type == "scatter":
            fig = px.scatter(filtered, x=x_col, y=y_col, title=title, template="plotly_dark", trendline="ols" if title=="Discount vs Sales" else None)
        elif chart_type == "heatmap":
            num = filtered.select_dtypes(include=np.number).corr()
            fig = px.imshow(num, text_auto=True, title=title, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------- MACHINE LEARNING ----------------
st.markdown("## 🤖 Sales Prediction Model (Random Forest)")
X = data[["Discount", "Profit"]]
y = data["Sales"]
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X, y)

col1, col2 = st.columns(2)
discount_in = col1.number_input("Discount %", min_value=0.0, max_value=50.0, value=5.0)
profit_in = col2.number_input("Profit", min_value=-500.0, max_value=1000.0, value=20.0)

pred = model.predict([[discount_in, profit_in]])[0]
st.success(f"📌 Predicted Sales: ₹{pred:,.2f}")
