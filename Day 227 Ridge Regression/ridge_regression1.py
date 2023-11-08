# Ridge Regression 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

# loading the datas
data = load_diabetes()

print(data.DESCR)

# splitting the datas into x and y
X = data.data
y = data.target

from sklearn.model_selection import train_test_split
x_train , X_test, y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=45)

# applying linear regression
from sklearn.linear_model import LinearRegression

L = LinearRegression()

# fitting the model
L.fit(x_train,y_train)

y_pred = L.predict(X_test)

# accuracy
from sklearn.metrics import r2_score,mean_squared_error

print("R2 score = ",r2_score(y_test,y_pred))
print("Means squared error = ",np.sqrt(mean_squared_error(y_test,y_pred)))

# applying ridge regression
from sklearn.linear_model import Ridge

R = Ridge(alpha=0.0001)

R.fit(x_train,y_train)

y_pred1 = R.predict(X_test)

# accuracy

print("R2 score = ",r2_score(y_test,y_pred1))
print("Means squared error = ",np.sqrt(mean_squared_error(y_test,y_pred1)))

m=100
x1 = 5 * np.random.rand(m,1)-2
x2 = 0.7 * x1 ** 2 - 2 * x1 +3 + np.random.randn(m,1)

plt.scatter(x1,x2)
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1,x2,alpha):
    model = Pipeline([
        ('poly_feats',PolynomialFeatures(degree=16)),
        ('ridge',Ridge(alpha=alpha))
    ])
    model.fit(x1,x2)
    return model.predict(x1)

alphas = [0,20,200]
cs = ['r','g','b']

plt.figure(figsize=(10,6))
plt.plot(x1,x2,'b+',label='Datapints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Datapoints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    X = np.column_stack((x1, x2))  # Combine x1 and x2 into a 2D array
    model.fit(X, x2)  # Fit the model using the combined 2D array and the target x2
    return model.predict(X)

alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Data points')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)

    # Plot
    plt.plot(np.sort(x1[:,0]), preds[np.argsort(x1[:,0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()