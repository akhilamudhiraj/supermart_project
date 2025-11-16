# Supermart Grocery Sales Dashboard

This repository contains a **Supermart  Streamlit Dashboard** built for **Supermart Grocery Sales Data Analytics**. It showcases sales trends, profits, discounts, and predictions using Python, Streamlit, and Plotly.
## Features

- Interactive **Streamlit dashboard** with neon theme
- **Sales and Profit KPIs** for quick insights
- Visualizations:
  - Sales by Category
  - Monthly Sales Trend
  - Correlation Heatmap
  - Top 10 Subcategories
  - Sales by Region
  - Profit vs Sales
  - Profit by Category
  - Monthly Profit Trend
  - Discount vs Sales
- **Random Forest model** for sales prediction
- Filters with **type-to-search** functionality
## Dataset

The project uses the **Supermart Grocery Sales - Retail Analytics Dataset**.  
Path in the project: `data/Supermart Grocery Sales - Retail Analytics Dataset (1).csv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akhilamudhiraj/supermart_project.git
Navigate into the project folder:

cd supermart_project
Create a virtual environment (if not already created):
python -m venv venv
Activate the virtual environment:

Windows (PowerShell):

.\venv\Scripts\Activate.ps1
Install the required libraries:

pip install -r requirements.txt
Run the dashboard:

streamlit run scripts/dashboard.py
## Usage

1. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     & venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

2. Run the Streamlit dashboard:
   ```bash
   streamlit run scripts/dashboard.py
   Open the dashboard in your browser at the URL displayed in the terminal (usually http://localhost:8501).
## Project Description

This project is a **Supermart Grocery Sales Dashboard** built using **Python, Streamlit, Plotly, and scikit-learn**.  
It provides interactive visualizations, KPIs, and a simple **Random Forest model** for sales prediction.

### Features:
- Neon Electric Blue themed Streamlit dashboard
- Interactive filters for Category, Sub-Category, Region, and Year
- Key Performance Indicators (Total Sales, Total Profit, Avg Discount)
- 9 different charts including:
  - Sales by Category
  - Monthly Sales Trend
  - Correlation Heatmap
  - Top 10 Subcategories
  - Sales by Region
  - Profit vs Sales
  - Profit by Category
  - Monthly Profit Trend
  - Discount vs Sales
- Simple machine learning model (Random Forest) to predict sales based on discount and profit
## License

This project is licensed under the **MIT License**.  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, as long as you include this license notice in all copies or substantial portions of the software.

