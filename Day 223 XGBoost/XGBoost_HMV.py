# XGBoost - Handling missing values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# creating a sample dataset with missing values
X = np.array([[1,2],
[3,np.nan],
[7,8],
[5,6],
[3,4],
[8,9]])

y = np.array([0,1,0,1,0,1])

# splitting the dataset into train and test
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# initializing an XGBoost classifier object
model = xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')

# training the model
model.fit(X_train,y_train)

# making predictions
y_pred = model.predict(X_test)

# calculating accuracy 
accuracy = accuracy_score(y_test,y_pred)

print(accuracy)