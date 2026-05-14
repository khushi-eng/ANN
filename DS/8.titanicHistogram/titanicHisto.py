# STEP 1: Import Required Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 2: Load Titanic Dataset

data = sns.load_dataset('titanic')

# STEP 3: Display First 5 Rows

print("FIRST 5 ROWS")
print(data.head())

# STEP 4: Check Dataset Information

print("\nDATASET SHAPE")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns)

# STEP 5: Check Missing Values

print("\nMISSING VALUES")
print(data.isnull().sum())

# ---------------------------------------------------
# VISUALIZATION 1
# Count Plot of Survived Passengers
# ---------------------------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x='survived', data=data)

plt.title("Survival Count")

plt.xlabel("Survival Status")

plt.ylabel("Number of Passengers")

# Custom labels for 0 and 1

plt.xticks([0,1], ['Not Survived', 'Survived'])

plt.show()

# ---------------------------------------------------
# VISUALIZATION 2
# Survival based on Gender
# ---------------------------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x='survived', hue='sex', data=data)

plt.title("Survival based on Gender")

plt.xlabel("Survival Status")

plt.ylabel("Count")

plt.xticks([0,1], ['Not Survived', 'Survived'])

plt.show()
# ---------------------------------------------------
# VISUALIZATION 3
# Survival based on Age
# ---------------------------------------------------
# ---------------------------------------------------
# HEATMAP : Age Based Survival Analysis
# ---------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create Age Groups

data['Age_Group'] = pd.cut(
    data['age'],
    bins=[0, 10, 20, 30, 40, 50, 60, 80],
    labels=['0-10', '10-20', '20-30',
            '30-40', '40-50', '50-60', '60-80']
)

# Calculate Survival Count for Each Age Group

heatmap_data = data.groupby('Age_Group')['survived'].mean().to_frame()

# Plot Heatmap

plt.figure(figsize=(6,5))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='YlGnBu'
)

plt.title("Age Based Survival Rate")

plt.xlabel("Survival Rate")

plt.ylabel("Age Group")

plt.show()
# ---------------------------------------------------
# VISUALIZATION 4
# Histogram of Fare Distribution
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(data['fare'],
         bins=30)

plt.title("Distribution of Fare")

plt.xlabel("Fare")

plt.ylabel("Number of Passengers")

plt.show()