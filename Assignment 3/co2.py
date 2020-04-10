import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('global_co2.csv')
X = df.iloc[219:, 0:1].values
y = df.iloc[219:, 1].values
#print(df)


from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r1 = r2_score(y,y_poly_pred)
print('\nAccuracy of CO2 production per Year:')
print(r1)
print('\n\nCO2 production in 2011:')
print(model.predict(poly_reg.fit_transform([[2011]])))
print('CO2 production in 2012:')
print(model.predict(poly_reg.fit_transform([[2012]])))
print('CO2 production in 2013:')
print(model.predict(poly_reg.fit_transform([[2013]])))


plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('CO2 Production per Year')
plt.xlabel('Year')
plt.ylabel('CO2 Production')
plt.show()
