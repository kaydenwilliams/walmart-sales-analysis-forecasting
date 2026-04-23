import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/walmart_cleaned.csv')

# Finding which store is making most sales
store_performance = df.groupby('Type').agg(
    Total_Sales=('Weekly_Sales', 'sum'),
    Avg_Weekly_Sales=('Weekly_Sales', 'mean'),
    Avg_Size=('Size', 'mean'),
    Store_Count=('Store', 'nunique')
).sort_values('Total_Sales', ascending=False).round(2)

pd.options.display.float_format = '{:,.2f}'.format
print(store_performance)

# Finding store growth rates
store_growth = df.groupby(['Type', 'Date']).agg(
    Total_Sales=('Weekly_Sales', 'sum')
)

print(store_growth.head(20))

# Graph to view sales by store types
store_growth_reset = store_growth.reset_index()

for store_type in ['A', 'B', 'C']:
    subset = store_growth_reset[store_growth_reset['Type'] == store_type]
    plt.plot(subset['Date'], subset['Total_Sales'], label=f'Type {store_type}')

plt.title('Weekly Sales by Store Type Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/sales_by_type_over_time.png')
plt.show()

# Checking holiday impact on sales
holiday_impact = df.groupby('IsHoliday').agg(
    Total_Sales=('Weekly_Sales', 'sum'),
    Avg_Weekly_Sales=('Weekly_Sales', 'mean')
)

print(holiday_impact)

# Checking markdown impact on sales
df['Has_Markdown'] = (df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1) > 0)

markdown_impact = df.groupby('Has_Markdown').agg(
    Total_Sales=('Weekly_Sales', 'sum'),
    Avg_Weekly_Sales=('Weekly_Sales', 'mean')
)

print(markdown_impact)

# Finding whether unemployment and fuel prices affect weekly sales
df[['Weekly_Sales', 'Unemployment', 'Fuel_Price']].corr()
print(df[['Weekly_Sales', 'Unemployment', 'Fuel_Price']].corr())

print("=== KEY FINDINGS ===")
print(f"Type A revenue share: 64.3% ($4.33B of $6.73B total)")
print(f"Holiday weekly sales lift: 7.13% ($17,035 vs $15,901)")
print(f"Unemployment correlation: -0.03")
print(f"Fuel Price correlation: -0.00")