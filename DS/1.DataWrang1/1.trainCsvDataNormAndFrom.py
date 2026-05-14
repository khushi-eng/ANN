# Import Required Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Dataset

df = pd.read_csv("train.csv")

# Display First 5 Rows

print("First 5 Rows:")
print(df.head(5))

# Check Dimensions of Dataset

print("\nDimensions of Dataset:")
print(df.shape)

# Display Column Names

print("\nColumn Names:")
print(df.columns)

# Check Data Types

print("\nData Types:")
print(df.dtypes)

# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary

print("\nStatistical Summary:")
print(df.describe())

# Dataset Information

print("\nDataset Information:")
print(df.info())

# ---------------------------------------------------
# Data Formatting / Type Conversion
# ---------------------------------------------------

df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].astype('category')

print("\nUpdated Data Types:")
print(df.dtypes)

# ---------------------------------------------------
# Convert Categorical Variables into Numerical
# ---------------------------------------------------

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

print("\nConverted Numerical Values:")
print(df[['Sex', 'Embarked']].head())

# ---------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------

df['Age'] = df['Age'].fillna(df['Age'].mean())

# ---------------------------------------------------
# Data Normalization
# ---------------------------------------------------

scaler = MinMaxScaler()

df[['Age', 'Fare']] = scaler.fit_transform(
    df[['Age', 'Fare']]
)

print("\nNormalized Data:")
print(df[['Age', 'Fare']].head())

# ---------------------------------------------------
# Final Dataset
# ---------------------------------------------------

print("\nFinal Dataset:")
print(df.head())