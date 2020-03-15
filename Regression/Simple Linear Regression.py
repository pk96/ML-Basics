## SIMPLE LINEAR REGRESSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data.csv')

X = df.iloc[:, :-1].values ## all columns except last column as feature matrix
y = df.iloc[:, 1].values ## dependent variable vector

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


""" SLR Algorithm performs feature scaling
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) ## dont need to fit the scaler as it is already fitted on the training set """


# Fitting SLR to training data
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

slr.fit(X_train, y_train)

# Predicting Test set results
y_pred = slr.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, slr.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, slr.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

