# XGBoost - speed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import time

# creating toy dataset
X, y = make_classification(n_samples=10000,n_features=200,random_state=42)

# splitting the datas into train and test
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# initializing the models
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False,eval_metric='logloss')

# training and evaluating using Sklearn gradient boosting model
start_gb = time.time()
gb_model.fit(X_train,y_train)
end_gb = time.time()
gb_time = end_gb - start_gb

gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test,gb_predictions)


# training and evaluating using XGBoost model
start_xgb = time.time()
xgb_model.fit(X_train, y_train)
end_xgb = time.time()
xgb_time = end_xgb - start_xgb

xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test,xgb_predictions)

print("GB time = ",gb_time)
print("GB accuracy = ",gb_accuracy)

print("XGB time = ",xgb_time)
print("XGB accuracy = ",xgb_accuracy)