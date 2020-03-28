# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:07:07 2020

@author: User
"""


import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Brain Weight vs Head Size (Training set)')
plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Brain Weight vs Head Size (Test set)')
plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.show()
print(regressor.score(X_test, y_test))
