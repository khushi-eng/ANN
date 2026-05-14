# Import Required Libraries
import pandas as pd
import numpy as np

# Column Names
columns = ['Sepal_Length', 'Sepal_Width',
           'Petal_Length', 'Petal_Width', 'Species']

# Load CSV File
df = pd.read_csv("iris.csv", names=columns)

print(df.head())

# Display First 5 Rows
print("FIRST 5 ROWS OF DATASET")
print(df.head())

# Group Dataset by Species
grouped = df.groupby('Species')

# Summary Statistics
print("\nMEAN")
print(grouped.mean())

print("\nMEDIAN")
print(grouped.median())

print("\nMINIMUM VALUES")
print(grouped.min())

print("\nMAXIMUM VALUES")
print(grouped.max())

print("\nSTANDARD DEVIATION")
print(grouped.std())