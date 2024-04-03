import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('seattle-weather.csv')
data =data.set_index('date')

X = data[['precipitation','temp_max','temp_min','wind']].values 
y = data['weather'].values 

def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LinearRegression:
    def __init__(self):
        self.coef_ = None  
        self.intercept_ = None  
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        return X_b.dot(np.concatenate([[self.intercept_], self.coef_]))
    
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


def mean_squared_error(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    squared_errors = (y_test - y_pred) ** 2
    mse = squared_errors.mean()
    return mse

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Test Values')
plt.ylabel('Predictions')
plt.title('Predictions vs. Test Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_pred, label='Predictions', color='blue')
plt.plot(y_test, label='Test Values', color='red')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Values vs. Predictions')
plt.legend()
plt.grid(True)
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='red', linestyle='--')
plt.grid(True)
plt.show()