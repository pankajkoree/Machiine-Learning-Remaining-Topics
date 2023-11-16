# Elastic Net Regression

# importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV

# reading the dataset
df = pd.read_csv('Hitters.csv')

df = df.dropna()

# splitting into X and y
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])

y = df["Salary"]

X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=42)

# fitting the models
enet_model = ElasticNet().fit(X_train, y_train)

# getting the coeffiecient
enet_model.coef_

# getting the intercept
enet_model.intercept_

enet_model.predict(X_train)[:10]

enet_model.predict(X_test)[:10]

y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

# r2 accuracy
r2_score(y_test,y_pred)

enet_cv_model = ElasticNetCV(cv = 10).fit(X_train,y_train)

# If we don't give the lambdas, what's the alpha?

enet_cv_model.alpha_

enet_cv_model.intercept_

enet_cv_model.coef_

# Let's create the final model according to optimum alpha.

enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)

# Let's now calculate the error for the test set using this final model.

y_pred = enet_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))