# 📌 Project: Boston House Price Prediction (App Brewery Bootcamp Project)

# 🔹 Standard Libraries
import pandas as pd
import numpy as np

# 🔹 Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 🔹 Machine Learning & Stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

# 🔹 Load Data
data = pd.read_csv('House_Price_Boston.csv', index_col=0)

# Inspect data
print(data.head())
print(data.tail())
print(f'NaN values: {data.isna().sum().sum()}')
print(f'Duplicated rows: {data.duplicated().sum()}')

# Drop missing values (if any)
data = data.dropna()

# Basic stats
print(data.shape)
print(data.describe())
print(data.columns)

# 🔹 Descriptive Stats
print("Average students per teacher:", data['PTRATIO'].mean())
print("Average home price:", data['PRICE'].mean())
print("CHAS values (0 = not near river, 1 = near river):", data['CHAS'].unique())
print("Max rooms:", data['RM'].max())
print("Min rooms:", data['RM'].min())

# 🔹 Visualization: Distribution Plots
selected = ['PRICE', 'RM', 'DIS', 'RAD']
for col in selected:
    sns.displot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 🔹 Bar Chart: River Proximity
river_bar = data['CHAS'].value_counts().rename({0: 'No', 1: 'Yes'})
px.bar(river_bar, title='Properties Next to the Charles River').show()

# 🔹 Pairplot: Relationships
sns.pairplot(data[['NOX', 'DIS', 'RM', 'PRICE', 'LSTAT']])
plt.suptitle('Relationship Between Features', y=1.02)
plt.show()

# 🔹 Jointplots
sns.jointplot(data=data, x='DIS', y='NOX', kind='scatter')
sns.jointplot(data=data, x='INDUS', y='NOX', kind='scatter')
sns.jointplot(data=data, x='LSTAT', y='RM', kind='scatter')
sns.jointplot(data=data, x='LSTAT', y='PRICE', kind='scatter')
sns.jointplot(data=data, x='RM', y='PRICE', kind='scatter')
plt.show()

# 🔹 Prepare Data for Modeling
X = data.drop(columns=['PRICE'])
y = data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# 🔹 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("R² on training data:", model.score(X_train, y_train))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# 🔹 Predictions and Residuals
pred_train = model.predict(X_train)
residuals = y_train - pred_train

# 🔹 Scatterplots for Actual vs Predicted
sns.scatterplot(x=y_train, y=pred_train, color='skyblue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

sns.scatterplot(x=pred_train, y=residuals, color='purple')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions")
plt.show()

# 🔹 Distribution of Residuals
print("Residual mean:", np.mean(residuals))
print("Residual skewness:", stats.skew(residuals))

sns.displot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.show()

# 🔹 Log Transformation of PRICE
print("Original PRICE skew:", stats.skew(data['PRICE']))

log_price = np.log(data['PRICE'])

sns.displot(log_price, kde=True, color='orange')
plt.title("Log-Transformed PRICE Distribution")
plt.xlabel("log(PRICE)")
plt.show()

# Log mapping
plt.scatter(data['PRICE'], log_price)
plt.xlabel("Original PRICE")
plt.ylabel("Log PRICE")
plt.title("Original vs Log PRICE")
plt.show()

# 🔹 Retrain with Log PRICE
y_log = log_price
X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.20, random_state=10)

log_model = LinearRegression()
log_model.fit(X_train, y_log_train)

print("Log Model R²:", log_model.score(X_train, y_log_train))

# 🔹 Residuals from Log Model
log_pred = log_model.predict(X_train)
log_residuals = y_log_train - log_pred

sns.scatterplot(x=y_log_train, y=log_pred)
plt.xlabel("Actual Log Prices")
plt.ylabel("Predicted Log Prices")
plt.title("Actual vs Predicted (Log Prices)")
plt.show()

sns.scatterplot(x=log_pred, y=log_residuals)
plt.xlabel("Predicted Log Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions (Log)")
plt.show()

# 🔹 Compare Model Performances
print("Original R² (test):", model.score(X_test, y_test))
print("Log R² (test):", log_model.score(X_test, y_log_test))

# 🔹 Estimate Average Property Price
property_stats = X.mean().to_frame().T
log_estimate = log_model.predict(property_stats)[0]
dollar_est = np.exp(log_estimate) * 1000

print(f"Estimated price of average property: ${dollar_est:,.2f}")

# 🔹 Custom Prediction
def predict_price(nr_rooms, next_to_river, student_ratio, distance, pollution, poverty):
    stats = property_stats.copy()
    stats['RM'] = nr_rooms
    stats['PTRATIO'] = student_ratio
    stats['DIS'] = distance
    stats['CHAS'] = 1 if next_to_river else 0
    stats['NOX'] = pollution
    stats['LSTAT'] = poverty
    
    log_prediction = log_model.predict(stats)[0]
    return np.exp(log_prediction) * 1000

# Example:
prediction = predict_price(
    nr_rooms=8,
    next_to_river=True,
    student_ratio=20,
    distance=5,
    pollution=data['NOX'].quantile(0.75),
    poverty=data['LSTAT'].quantile(0.25)
)

print(f"Predicted custom property value: ${prediction:,.2f}")
