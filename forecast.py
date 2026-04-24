import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv('outputs/walmart_forecast.csv')
df = df.sample(n=50000, random_state=42)
# Engineer date features
df['Date'] = pd.to_datetime(df['Date'])
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# Define features and target
features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price',
'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
'CPI', 'Unemployment', 'Size', 'Week', 'Month', 'Year']
X = df[features]
y = df['Weekly_Sales']
# Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'outputs/walmart_model.pkl')
# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")
feature_importance = pd.Series(model.feature_importances_, index=features)
feature_importance.sort_values(ascending=False).plot(kind='bar')
print(feature_importance.sort_values(ascending=False))
plt.title('Feature Importance — Walmart Sales Forecast')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.show()