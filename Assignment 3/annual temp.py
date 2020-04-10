
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
# Importing the dataset
df = pd.read_csv('annual_temp.csv')
gcag=df.loc[(df.Source=='GCAG')]
gis=df.loc[(df.Source=='GISTEMP')]
#print(gis)
#print(gcag)

X1 = gis.iloc[:, 1:2].values
y1 = gis.iloc[:, 2].values

X2 = gcag.iloc[:, 1:2].values
y2 = gcag.iloc[:, 2].values



from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X1)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_poly, y1)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y1,y_poly_pred))
r1 = r2_score(y1,y_poly_pred)
print("Accuracy score for GISTEMP dataset:")
print(r1)

gis16=model.predict(poly_reg.fit_transform([[2016]]))
gis17=model.predict(poly_reg.fit_transform([[2017]]))


plt.scatter(X1, y1, color = 'red')
plt.plot(X1, model.predict(poly_reg.fit_transform(X1)), color = 'blue')
plt.title('GIS TEMP')
plt.xlabel('Year')
plt.ylabel('Mean')
plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=2)
x_poly2 = poly_reg.fit_transform(X2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_poly2, y2)
y_poly_pred2 = model.predict(x_poly2)

rmse = np.sqrt(mean_squared_error(y2,y_poly_pred2))
r2 = r2_score(y2,y_poly_pred2)
print("Accuracy score for GCAG dataset:")
print(r2)


gcag16=model.predict(poly_reg.fit_transform([[2016]]))
gcag17=model.predict(poly_reg.fit_transform([[2017]]))


plt.scatter(X2, y2, color = 'red')
plt.plot(X2, model.predict(poly_reg.fit_transform(X2)), color = 'blue')
plt.title('GCAG')
plt.xlabel('Year')
plt.ylabel('Mean')
plt.show()


print('\n\nTemperature in both industries in 2016..')
print('GCAG: ')
print(gcag16)
print('GISTEMP: ')
print(gis16)


print('\n\nTemperature in both industries in 2017..')
print('GCAG: ')
print(gcag17)
print('GISTEMP: ')
print(gis17)
