# SIMPLE LINEAR REGRESSION

df = read.csv('Salary_Data.csv')


# Splitting dataset into training and testing set
#install.packages('caTools')
library('caTools')
set.seed(123)
split = sample.split(df$Salary, SplitRatio = 2/3)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)


# Feature Scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Regression model to training data
slr = lm(formula = Salary ~ YearsExperience,
         data = training_set)
summary(slr)

# Predicting test set results
y_pred = predict(slr, newdata = test_set)


# Visualizing Training set results
#install.packages('ggplot2')
library('ggplot2')

ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(slr, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs. Experience (Training Set)') + 
  xlab('Years of Experience') +
  ylab('Salary') 

# Visualizing Test set results
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(slr, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs. Experience (Test Set)') + 
  xlab('Years of Experience') +
  ylab('Salary') 
