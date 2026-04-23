import pandas as pd

# Load all 3 files
train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

# Fix the date columns from string to date
train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])

# Filling the missing markdowns with 0 (no markdown even those weeks)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 
                 'MarkDown4', 'MarkDown5']
features[markdown_cols] = features[markdown_cols].fillna(0)

# Forward fill CPI and Unemployment (no new reading yet)
features['CPI'] = features['CPI'].ffill()
features['Unemployment'] = features['Unemployment'].ffill()

# Merge train with features on Store and Date
df = pd.merge(train, features, on=['Store', 'Date'], how='left')

# Merge result with stores on Store
df = pd.merge(df, stores, on='Store', how='left')

# Drop duplicate IsHoliday column and rename the remaining one
df = df.drop(columns=['IsHoliday_y'])
df = df.rename(columns={'IsHoliday_x': 'IsHoliday'})

# Verify the merge
print("Merged shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values after merge:")
print(df.isnull().sum())

# Check for negative sales
negative_sales = df[df['Weekly_Sales'] < 0]
print(f"\nNegative sales rows: {len(negative_sales)}")
print(f"Total negative sales value: ${negative_sales['Weekly_Sales'].sum():,.2f}")

df['Has_Markdown'] = (df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1) > 0)

# Keep full dataset for revenue analysis
df.to_csv('outputs/walmart_cleaned.csv', index=False)

# Separate forecasting dataset, purchases only
df_forecast = df[df['Weekly_Sales'] >= 0]
df_forecast.to_csv('outputs/walmart_forecast.csv', index=False)

print(f"Full dataset: {len(df)} rows")
print(f"Forecast dataset: {len(df_forecast)} rows")
print(f"Removed {len(df) - len(df_forecast)} negative sales rows for forecasting")

print("\nCleaned dataset saved.")