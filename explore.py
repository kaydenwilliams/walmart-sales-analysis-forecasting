import pandas as pd

train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

print("=== TRAIN ===")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())
print("Missing values:")
print(train.isnull().sum())
print("\nFirst 5 rows:")
print(train.head())

print("\n=== FEATURES ===")
print("Shape:", features.shape)
print("Columns:", features.columns.tolist())
print("Missing values:")
print(features.isnull().sum())
print("\nFirst 5 rows:")
print(features.head())

print("\n=== STORES ===")
print("Shape:", stores.shape)
print("Columns:", stores.columns.tolist())
print("\nFirst 5 rows:")
print(stores.head())
print("\nStore types:")
print(stores['Type'].value_counts())