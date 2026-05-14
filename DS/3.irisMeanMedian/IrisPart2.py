# Import Required Library
import pandas as pd



# Column Names
columns = ['Sepal_Length', 'Sepal_Width',
           'Petal_Length', 'Petal_Width', 'Species']

# Load CSV File
df = pd.read_csv("iris.csv", names=columns)

print(df.head())

# Display Dataset Information
print("FIRST 5 ROWS")
print(df.head())

# Group by Species
species_group = df.groupby('Species')

# Display Statistical Details
for species, data in species_group:

    print("\n===================================")
    print("SPECIES :", species)
    print("===================================")

    print("\nMEAN")
    print(data.mean(numeric_only=True))

    print("\nSTANDARD DEVIATION")
    print(data.std(numeric_only=True))

    print("\nPERCENTILES")
    print(data.describe())