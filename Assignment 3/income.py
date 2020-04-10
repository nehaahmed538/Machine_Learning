
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('monthlyexp vs incom.csv')
X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values
#print(df)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 1/3, random_state = 1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


poly_features = PolynomialFeatures(degree=7)

X_train_poly = poly_features.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

y_train_predicted = poly_model.predict(X_train_poly)

y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
r2_train = r2_score(Y_train, y_train_predicted)

rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
r2_test = r2_score(Y_test, y_test_predict)

print("\nTraining set score: " )
print(r2_train)
print("Test set score: ")
print(r2_test)


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_grid, poly_model.predict(poly_features.fit_transform(X_grid)), color = 'blue')
plt.title('Experience Vs Monthly Income (TRAIN SET)')
plt.xlabel('Experience')
plt.ylabel('Monthly Income')
plt.show()


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_grid, poly_model.predict(poly_features.fit_transform(X_grid)), color = 'blue')
plt.title('Experience Vs Monthly Income (TEST SET)')
plt.xlabel('Experience')
plt.ylabel('Monthly Income')
plt.show()



