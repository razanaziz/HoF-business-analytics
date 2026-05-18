import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load customer satisfaction dataset
df = pd.read_excel("customer_satisfaction.xlsx")

# Check for missing values
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Composite customer satisfaction score
df['Composite_Score'] = (
    df['Excellent'] * 5 +
    df['Good'] * 4 +
    df['Average'] * 3 +
    df['Poor'] * 2 +
    df['Unsatisfactory'] * 1
)

# Normalize scores
scaler = MinMaxScaler()
df['Scaled_Score'] = scaler.fit_transform(
    df[['Composite_Score']]
)

print(df.head())
