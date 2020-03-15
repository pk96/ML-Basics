## MULTIPLE LINEAR REGRESSION

data = read.csv('50_Startups.csv')

# Encoding categorical data
data$State = factor(data$State,
                      levels = c('New York', 'California', 'Florida'),
                      labels = c(1, 2, 3))


# Splitting dataset into training and testing set
#install.packages('caTools')
library('caTools')
set.seed(123)
split = sample.split(data$Profit, SplitRatio = 0.8)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# Fitting MLR to the training set
mlr = lm(formula = Profit ~ .,
         data = training_set)
summary(mlr)

# Predicting Test set results
y_pred = predict(mlr, newdata = test_set)

# Backwards Elimination
mlr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
         data = data)
summary(mlr)

mlr = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
         data = data)
summary(mlr)

mlr = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
         data = data)
summary(mlr)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = data[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
