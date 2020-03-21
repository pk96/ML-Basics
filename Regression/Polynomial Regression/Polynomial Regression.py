## POLYNOMIAL REGRESSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values ## all columns except last column as feature matrix
y = df.iloc[:, 2].values ## dependent variable vector

# Splitting into training and testing set is not required here as the number of data points is small 
# and we know that we want to predict the salary of an emplyee of level 6.5 

# Fitting Linear Regression model to data
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

slr.fit(X, y)

# Fitting Polynomial Regression to data
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4) #this transforms the matrix of features X to X_poly. [x1] -> [x1, x1^2, x1^3, ... , x1^n]

X_poly = poly.fit_transform(X)

plr = LinearRegression()
plr.fit(X_poly, y)

# Visualizing Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, slr.predict(X), color = 'blue')
plt.title('Employee Level vs. Salary (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, plr.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('Employee Level vs. Salary (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting new result with Linear Regression
print('Salary for employee level 6.5 (Linear Regression):', slr.predict([[6.5]]))

# Predicting new result with Polynomial Regression
print('Salary for employee level 6.5 (Polynomial Regression):', plr.predict(poly.fit_transform([[6.5]])))
