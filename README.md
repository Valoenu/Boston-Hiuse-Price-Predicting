This project predicts Boston housing prices using Linear Regression, Exploratory Data Analysis (EDA), and log transformations. It includes data cleaning, visualizations, performance metrics, and a custom property value estimator.

📘 This project was developed as part of The App Brewery’s Python Bootcamp course to apply real-world data science and machine learning concepts.

⸻

📌 Project Goals
	•	Perform exploratory data analysis on Boston housing data
	•	Visualize feature relationships using Seaborn, Matplotlib & Plotly
	•	Build and evaluate a linear regression model
	•	Apply log transformation to improve accuracy
	•	Predict house values based on customizable property inputs

⸻

📂 Dataset
	•	Source: Boston Housing Dataset
	•	Provided by: App Brewery Machine Learning Section
	•	Features:
	•	PRICE: Median value of owner-occupied homes
	•	RM: Number of rooms
	•	DIS: Distance to employment centers
	•	NOX: Nitric oxides concentration (pollution)
	•	LSTAT: % of lower status population
	•	CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
	•	…and others

⸻

🛠️ Tools & Libraries
	•	Python
	•	pandas, numpy – data handling
	•	seaborn, matplotlib, plotly – data visualization
	•	sklearn – model training & evaluation
	•	scipy.stats – statistics (skewness, residuals)

⸻

📊 Visualizations
	•	Distributions of target & features
	•	Pairplots & jointplots showing relationships
	•	Residual analysis plots
	•	Log-transformed distribution comparisons
	•	Interactive bar charts with Plotly

⸻

📈 Model Summary
	•	Model: Linear Regression
	•	Target: PRICE
	•	Performance Metric: R² Score
	•	Includes both original and log-transformed models to evaluate improvements.

⸻

🏘️ Custom Property Estimator

At the end of the notebook, you can input values such as:
	•	Rooms
	•	Distance to city center
	•	Student-teacher ratio
	•	Pollution level
	•	Charles River proximity
	•	Poverty levels

And receive a real-time estimated property price.

⸻

📚 What I Learned
	•	How to explore and visualize housing data
	•	Why skewed data can affect model performance
	•	When and how to apply log transformation
	•	How to interpret regression coefficients and residuals
