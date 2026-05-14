# STEP 1: Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 2: Load Dataset

data = pd.read_csv("Social_Network_Ads.csv")

# STEP 3: Display First 5 Rows

print("FIRST 5 ROWS")
print(data.head())

# STEP 4: Check Missing Values

print("\nMISSING VALUES")
print(data.isnull().sum())

# STEP 5: Convert Categorical Data into Numerical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])

print("\nDATA AFTER LABEL ENCODING")
print(data.head())

# STEP 6: Select Independent and Dependent Variables

# Features
x = data[['Age', 'EstimatedSalary']]

# Target
y = data['Purchased']

# STEP 7: Split Dataset into Training and Testing

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=0
)

# STEP 8: Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# STEP 9: Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(xtrain, ytrain)

print("\nMODEL TRAINED SUCCESSFULLY")

# STEP 10: Predict Test Data

ypred = model.predict(xtest)

print("\nPREDICTED VALUES")
print(ypred)

# STEP 11: Compare Actual and Predicted Values

comparison = pd.DataFrame({
    'Actual': ytest,
    'Predicted': ypred
})

print("\nACTUAL VS PREDICTED")
print(comparison.head(10))

# STEP 12: Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, ypred)

print("\nCONFUSION MATRIX")
print(cm)

# STEP 13: Extract TP, TN, FP, FN

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTRUE POSITIVE =", TP)
print("TRUE NEGATIVE =", TN)
print("FALSE POSITIVE =", FP)
print("FALSE NEGATIVE =", FN)

# STEP 14: Calculate Accuracy, Error Rate,
# Precision and Recall

accuracy = (TP + TN) / (TP + TN + FP + FN)

error_rate = (FP + FN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

print("\nACCURACY =", accuracy)

print("ERROR RATE =", error_rate)

print("PRECISION =", precision)

print("RECALL =", recall)