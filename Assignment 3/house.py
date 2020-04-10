
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('housing price.csv')
X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values
#print(df)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('House ID VS Pricing')
plt.xlabel('House ID')
plt.ylabel('Pricing')
plt.show()

