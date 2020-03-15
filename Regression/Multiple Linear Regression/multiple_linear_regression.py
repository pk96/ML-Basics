## MULTIPLE LINEAR REGRESSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('50_startups.csv')

X = df.iloc[:, :-1].values ## all columns except last column as feature matrix
y = df.iloc[:, 4].values ## dependent variable vector

# Encoding State Column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoid dummy variable trap
X = X[:, 1:]

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting MLR to the training set
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

mlr.fit(X_train, y_train)

#Predicting Test set results
y_pred = mlr.predict(X_test)

# Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) #adding a column of 1's to act as X0 coefficient for B0 in the equation

X_opt = X[:, [0, 1, 2, 3, 4, 5]] #initialising the optimal matrix of features to include all columns at first 

"""mlr_ols = sm.OLS(endog = y, exog = X_opt).fit()
mlr_ols.summary()

X_opt = X[:, [0, 1, 3, 4, 5]] 
mlr_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(mlr_ols.summary())

X_opt = X[:, [0, 3, 4, 5]] 
mlr_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(mlr_ols.summary())

X_opt = X[:, [0, 3, 5]] 
mlr_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(mlr_ols.summary()) """

# Backward elimination with p-value and Adjusted R squared
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print (regressor_OLS.summary() )
    return x

SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)