# XGBoost Day 4

# XGBoost for regression
import numpy as np
import pandas as pd
from matplotlib import pyplot
import xgboost

# load dataset

data = pd.read_csv(r'D:\\copy_of_htdocs\\practice\\Python\\300days\\Day 246 XGBoost day 4\\housing.csv')

data

data.shape

X = data.iloc[:, :-1] 
y = data.iloc[:, -1]   

import xgboost
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold,cross_val_score

...
# define model
model = XGBRegressor()

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )