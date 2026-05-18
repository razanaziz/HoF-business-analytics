import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_excel("sales_regression.xlsx")

# Independent variables
X = df[['Ad_Spend', 'Brand_Recognition', 'Satisfaction_Score']]

# Dependent variable
y = df['Sales_Revenue']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, predictions))

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
