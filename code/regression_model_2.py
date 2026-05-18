import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load financial dataset
df = pd.read_excel("financial_data.xlsx")

# Features
X = df[
    [
        'Total_Revenue',
        'COGS',
        'Operating_Expenses',
        'Debt_Level',
        'Interest_Expense'
    ]
]

# Target
y = df['Net_Profit_Margin']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
print("MSE:", mean_squared_error(y_test, predictions))
print("R²:", model.score(X_test, y_test))
