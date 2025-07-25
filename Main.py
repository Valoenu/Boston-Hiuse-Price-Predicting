# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 📂 Load Dataset
# Make sure House_Price_Boston.csv is in the same directory as this script
data = pd.read_csv('House_Price_Boston.csv', index_col=0)

# 🧼 Inspect & Clean Data
print(data.head())                          # Preview top rows
print(data.tail())                          # Preview bottom rows
print(data.isna().sum().sum())              # Count missing values
print(data.duplicated().sum())              # Count duplicates
data.dropna(inplace=True)                   # Drop missing values (if any)

# 🔍 Quick Stats
print(data.describe())
print(f"Dataset shape: {data.shape}")
print(f"Average PTRATIO: {data['PTRATIO'].mean():.2f}")
print(f"Average house price: ${data['PRICE'].mean():.2f}")
print(f"Min CHAS: {data['CHAS'].min()}, Max CHAS: {data['CHAS'].max()}")
print(f"Rooms per dwelling → min: {data['RM'].min()}, max: {data['RM'].max()}")

# 📊 Visualize Distributions
for col in ['PRICE', 'RM', 'DIS', 'RAD']:
    sns.displot(data[col], kde=True, aspect=2)
    plt.title(f'Distribution of {col}')
    plt.show()

# 🏞 Properties near Charles River
chas_count = data['CHAS'].value_counts().reset_index()
chas_count.columns = ['CHAS', 'Count']
chas_count['CHAS'] = chas_count['CHAS'].map({0: 'No', 1: 'Yes'})
px.bar(chas_count, x='CHAS', y='Count', title='Next to Charles River').show()

# 🔗 Relationships between variables
sns.pairplot(data[['NOX', 'DIS', 'RM', 'PRICE', 'LSTAT']])
plt.suptitle("Pairplot: Key Features", y=1.02)
plt.show()

# 🧠 Key Jointplots
sns.jointplot(data=data, x='DIS', y='NOX', kind='scatter')
sns.jointplot(data=data, x='INDUS', y='NOX', kind='scatter')
sns.jointplot(data=data, x='LSTAT', y='RM', kind='scatter')
sns.jointplot(data=data, x='LSTAT', y='PRICE', kind='scatter')
sns.jointplot(data=data, x='RM', y='PRICE', kind='scatter')
plt.show()

# 📈 Feature/Target split
X = data.drop(columns=['PRICE'])
y = data['PRICE']

# 🎲 Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 🧠 Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📋 Results
print(f"Model R² score: {model.score(X_train, y_train):.3f}")
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 🔍 Predictions and residuals
y_pred = model.predict(X_train)
residuals = y_train - y_pred

# 🎯 Actual vs Predicted
sns.scatterplot(x=y_train, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# 📉 Residuals Plot
sns.scatterplot(x=y_pred, y=residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# 📊 Residual Distribution
sns.displot(residuals, kde=True, aspect=2)
plt.title("Residuals Distribution")
plt.show()

# ➕ Check skewness
print(f"Residual skewness: {stats.skew(residuals):.3f}")

# 🔁 Log-transform the target to reduce skew
log_y = np.log(y)

# 📊 Visualize log-transformed target
sns.displot(log_y, kde=True, aspect=2, color='orange')
plt.title("Log-Transformed Target: PRICE")
plt.show()

# 🚀 Train model on log-transformed target
X_train, X_test, log_y_train, log_y_test = train_test_split(X, log_y, test_size=0.2, random_state=10)
log_model = LinearRegression()
log_model.fit(X_train, log_y_train)

# 📊 Evaluate log-model
log_pred = log_model.predict(X_train)
log_residuals = log_y_train - log_pred
print(f"Log-model R² score: {log_model.score(X_train, log_y_train):.3f}")

# 🎯 Residuals for log-model
sns.scatterplot(x=log_pred, y=log_residuals)
plt.xlabel("Predicted log(PRICE)")
plt.ylabel("Residuals")
plt.title("Residuals of Log-Model")
plt.show()

# 🏡 Estimate Price for Average Property
mean_features = X.mean().values.reshape(1, -1)
log_estimate = log_model.predict(mean_features)[0]
dollar_estimate = np.exp(log_estimate) * 1000
print(f"Estimated average house value: ${dollar_estimate:,.2f}")

# 🧠 Estimate Price for Custom Property
property_stats = pd.DataFrame(mean_features, columns=X.columns)
property_stats['CHAS'] = 1               # Close to Charles River
property_stats['RM'] = 8                 # More rooms
property_stats['PTRATIO'] = 20           # Reasonable student-teacher ratio
property_stats['DIS'] = 5                # Moderate distance to employment
property_stats['NOX'] = data['NOX'].quantile(0.75)  # Higher pollution
property_stats['LSTAT'] = data['LSTAT'].quantile(0.25)  # Low % lower status

log_estimate = log_model.predict(property_stats)[0]
dollar_estimate = np.exp(log_estimate) * 1000
print(f"Estimated value for custom property: ${dollar_estimate:,.2f}")
