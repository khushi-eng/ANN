# STEP 1: Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 2: Load Dataset from CSV File

data = pd.read_csv("Boston_Housing_Data.csv")

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

# STEP 6: Separate Features and Target Variable

# Independent Variables
x = data.drop(['MEDV'], axis=1)

# Dependent Variable
y = data['MEDV']

# STEP 7: Split Dataset into Training and Testing

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

print("\nTRAINING DATA SIZE")
print(xtrain.shape)

print("\nTESTING DATA SIZE")
print(xtest.shape)

# STEP 8: Create Linear Regression Model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# Train Model
model = lm.fit(xtrain, ytrain)

print("\nMODEL TRAINED SUCCESSFULLY")

# STEP 9: Predict House Prices

ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)

# STEP 10: Compare Actual and Predicted Prices

comparison = pd.DataFrame({
    'Actual Price': ytest,
    'Predicted Price': ytest_pred
})

print("\nACTUAL VS PREDICTED VALUES")
print(comparison.head(10))

# STEP 11: Evaluate Model Performance

from sklearn.metrics import mean_squared_error, r2_score

# Mean Squared Error
mse = mean_squared_error(ytest, ytest_pred)

print("\nMEAN SQUARED ERROR")
print(mse)

# R2 Score
r2 = r2_score(ytest, ytest_pred)

print("\nR2 SCORE")
print(r2)

# STEP 12: Plot Graph

plt.scatter(ytrain,
            ytrain_pred,
            c='blue',
            marker='o',
            label='Training data')

plt.scatter(ytest,
            ytest_pred,
            c='lightgreen',
            marker='s',
            label='Testing data')

plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.title('Actual vs Predicted House Prices')

plt.legend(loc='upper left')

plt.show()