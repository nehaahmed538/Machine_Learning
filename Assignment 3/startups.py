
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x_test=np.array([165349.2 , 136897.8 , 471784.1])
df = pd.read_csv('50_Startups.csv')
cal=df.loc[(df.State=='California')]
X1 = cal.iloc[:, 0:3].values
y1 = cal.iloc[:, 4].values
#print(cal)
ny=df.loc[(df.State=='New York')]
X2 = ny.iloc[:, 0:3].values
y2 = ny.iloc[:, 4].values
#print(ny)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X1, y1)

y_pred1 = reg.predict([[243838 , 393454 , 432654]])


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X2, y2)
y_pred2 = reg.predict([[243838 , 393454 , 432654]])

print('Accuracy score for California dataset:')
print(reg.score(X1, y1))
print('Accuracy score for New York dataset:')
print(reg.score(X2, y2))

print('\n \nPredicting profit in both states on same data...\n')

print('Profit obtained in California:')
print(y_pred1)
print('Profit obtained in New York:')
print(y_pred2)


