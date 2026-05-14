# STEP 1: Import Required Libraries

import numpy as np
import pandas as pd

# STEP 2: Load Dataset

data = pd.read_csv("Iris.csv")

# STEP 3: Remove Id Column

data = data.drop('Id', axis=1)

# STEP 4: Display First 5 Rows

print("FIRST 5 ROWS")
print(data.head())

# STEP 5: Check Missing Values

print("\nMISSING VALUES")
print(data.isnull().sum())

# STEP 6: Convert Species into Numerical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Species'] = le.fit_transform(data['Species'])

print("\nDATA AFTER LABEL ENCODING")
print(data.head())

# STEP 7: Select Features and Target

x = data[['SepalLengthCm',
          'SepalWidthCm',
          'PetalLengthCm',
          'PetalWidthCm']]

y = data['Species']

# STEP 8: Split Dataset

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=0
)

# STEP 9: Train Naive Bayes Model

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(xtrain, ytrain)

print("\nMODEL TRAINED SUCCESSFULLY")

# STEP 10: Predict

ypred = model.predict(xtest)

print("\nPREDICTED VALUES")
print(ypred)

# STEP 11: Show Input Features with
# Actual and Predicted Output

comparison = xtest.copy()

comparison['Actual Species'] = ytest.values

comparison['Predicted Species'] = ypred

print("\nACTUAL VS PREDICTED")
print(comparison.head(10))

# STEP 12: Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, ypred)

print("\nCONFUSION MATRIX")
print(cm)

# STEP 13: Accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ytest, ypred)

print("\nACCURACY =", accuracy)

# STEP 14: Classification Report

from sklearn.metrics import classification_report

print("\nCLASSIFICATION REPORT")
print(classification_report(ytest, ypred))