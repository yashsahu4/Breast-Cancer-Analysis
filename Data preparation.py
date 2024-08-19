# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()

# Convert to Pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column
df['target'] = data.target

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Features (all columns except 'target')
X = df.drop('target', axis=1)

# Target variable
y = df['target']

print("Features:\n", X.head())
print("Target:\n", y.head())


