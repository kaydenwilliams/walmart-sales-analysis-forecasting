# Walmart Store Sales Analysis

## Overview
Analysis of 421,570 weekly sales records across 45 Walmart stores from 2010 to 2012.
Identifies the impact of store type, holidays, markdowns, and external economic factors on sales performance.
Includes a machine learning forecasting model built with scikit-learn.

## Tools
Python, pandas, scikit-learn, matplotlib, MySQL, Tableau

## Dataset
Walmart Recruiting Store Sales Forecasting - Kaggle
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

## Key Findings
- Type A stores (avg 182K sq ft) generated 64.3% of total revenue ($4.33B of $6.73B), confirming store size as the primary driver of sales volume
- Holiday weeks averaged 7.13% higher weekly sales than non-holiday weeks ($17,035 vs $15,901)
- Promotional markdowns produced only a 1.92% average weekly sales lift ($16,177 vs $15,872), suggesting markdowns may not justify their margin cost
- Unemployment (r = -0.03) and fuel price (r = -0.00) showed near-zero correlation with weekly sales, confirming Walmart's revenue is largely insulated from macroeconomic conditions

## Files
- `clean.py` — data cleaning and merging of train, features, and stores datasets
- `eda.py` — exploratory data analysis and business question analysis
- `forecast.py` — machine learning sales forecasting model
- `outputs/walmart_cleaned.csv` — full cleaned dataset (421,570 rows)
- `outputs/walmart_forecast.csv` — dataset with negatives removed for modeling
- `executive_summary.pdf` — plain English summary of findings for non-technical stakeholders

## Tableau Dashboard
https://public.tableau.com/app/profile/kayden.williams2622/viz/Book1_17769827495830/WalmartSalesAnalysis#1

## Business Recommendations
1. Prioritize investment in Type A store expansion, they generate 2x the average weekly revenue of Type B stores
2. Reassess markdown strategy, less than 2% sales lift suggests promotional discounts are not driving meaningful revenue gains
3. Holiday staffing and inventory should be concentrated in Type A and B stores where holiday spikes are most pronounced
4. External economic conditions (unemployment, fuel prices) should not be primary inputs for store-level sales planning
