# =========================================================
# ASSIGNMENT 2 : DATA WRANGLING II
# =========================================================

# Problem Statement:
# Create an Academic Performance dataset and perform:
# 1. Missing value handling
# 2. Inconsistency detection and correction
# 3. Outlier detection and treatment
# 4. Data transformation
# =========================================================

# ---------------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 1 : Create Academic Performance Dataset
# ---------------------------------------------------------

data = {
    'Student_ID': [101,102,103,104,105,106,107,108,109,110],

    'Maths': [78, 85, np.nan, 92, 35, 150, 88, 76, 95, -5],

    'Science': [67, 92, 81, np.nan, 45, 85, 79, 98, 110, 84],

    'English': [72, 88, 79, 93, 40, 87, np.nan, 77, 90, 86],

    'Attendance': [85, 90, 78, 95, 60, 88, 92, 80, 105, -10]
}

df = pd.DataFrame(data)

print("=================================================")
print("Original Dataset")
print("=================================================\n")

print(df)

# ---------------------------------------------------------
# STEP 2 : Check Missing Values
# ---------------------------------------------------------

print("\n=================================================")
print("Missing Values")
print("=================================================\n")

print(df.isnull().sum())

# ---------------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------------

# Replace missing values using mean

numeric_columns = ['Maths', 'Science', 'English', 'Attendance']

for column in numeric_columns:

    mean_value = df[column].mean()

    df[column].fillna(mean_value, inplace=True)

print("\nDataset after handling missing values:\n")

print(df)

# ---------------------------------------------------------
# STEP 3 : Detect and Handle Inconsistencies
# ---------------------------------------------------------

print("\n=================================================")
print("Inconsistency Detection")
print("=================================================\n")

subject_columns = ['Maths', 'Science', 'English']

# Marks should be between 0 and 100

for column in subject_columns:

    invalid_values = df[
        (df[column] < 0) | (df[column] > 100)
    ]

    if not invalid_values.empty:

        print(f"\nInvalid values found in {column}:")

        print(invalid_values[[column]])

    # Replace values less than 0 with 0
    df.loc[df[column] < 0, column] = 0

    # Replace values greater than 100 with 100
    df.loc[df[column] > 100, column] = 100

# Attendance should be between 0 and 100

invalid_attendance = df[
    (df['Attendance'] < 0) | (df['Attendance'] > 100)
]

if not invalid_attendance.empty:

    print("\nInvalid values found in Attendance:")

    print(invalid_attendance[['Attendance']])

# Correct attendance values

df.loc[df['Attendance'] < 0, 'Attendance'] = 0

df.loc[df['Attendance'] > 100, 'Attendance'] = 100

print("\nDataset after handling inconsistencies:\n")

print(df)

# ---------------------------------------------------------
# STEP 4 : Detect and Handle Outliers using IQR Method
# ---------------------------------------------------------

print("\n=================================================")
print("Outlier Detection")
print("=================================================\n")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)

    upper_limit = Q3 + (1.5 * IQR)

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nOutliers in {column}:")

    print(outliers[[column]])

    # Replace outliers with median

    median_value = df[column].median()

    df.loc[df[column] < lower_limit, column] = median_value

    df.loc[df[column] > upper_limit, column] = median_value

print("\nDataset after Outlier Treatment:\n")

print(df)

# ---------------------------------------------------------
# STEP 5 : Data Transformation
# ---------------------------------------------------------

print("\n=================================================")
print("Data Transformation")
print("=================================================\n")

# Apply Min-Max Scaling on Attendance

scaler = MinMaxScaler()

df['Attendance_Scaled'] = scaler.fit_transform(
    df[['Attendance']]
)

print(df)

# ---------------------------------------------------------
# STEP 6 : Visualization
# ---------------------------------------------------------

plt.hist(df['Attendance_Scaled'])

plt.title("Scaled Attendance Distribution")

plt.xlabel("Scaled Attendance")

plt.ylabel("Frequency")

plt.show()

# ---------------------------------------------------------
# FINAL DATASET
# ---------------------------------------------------------

print("\n=================================================")
print("Final Cleaned Dataset")
print("=================================================\n")

print(df)