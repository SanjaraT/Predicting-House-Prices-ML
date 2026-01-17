-->Overview

This project predicts house prices using the Kaggle Housing Prices dataset. The task is formulated as a regression problem with a log-transformed target variable to handle skewness.

-->Dataset

Source: Kaggle – Housing Prices Dataset
Target: price (log-transformed using log1p)
Features: Numerical + categorical housing attributes

-->Preprocessing

Checked missing & duplicate values
Encoded categorical features
Log-transformed target (price)
Train / validation / test split (70/15/15)
Feature scaling for linear models

--> Models

Linear Regression
Ridge & Lasso (GridSearchCV)
Random Forest
Gradient Boosting

-->Results 

Test (Gradient Boosting):

RMSE: 0.247
R²: 0.651

-->Conclusion

Gradient Boosting achieved the best performance, while linear models remained strong baselines. The final results are realistic and free from data leakage.