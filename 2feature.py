
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def update_weights(X, y, weights, learning_rate):
    y_pred = np.dot(X, weights)
    error = y - y_pred
    gradient = -2 * np.dot(X.T, error) / len(X)
    weights -= learning_rate * gradient
    return weights

def perceptron_mse(X, y, learning_rate, rounds, initial_weights=None):
    if initial_weights is None:
        weights = np.zeros(X.shape[1], dtype='float64')
    else:
        weights = initial_weights.astype('float64')

    y = y.astype('float64')

    for epoch in range(rounds):
        weights = update_weights(X, y, weights, learning_rate)
        y_pred = np.dot(X, weights)
        mse = mean_squared_error(y, y_pred)
        print(f'Epoch {epoch+1}/{rounds}, MSE: {mse}')
    return weights


data = pd.read_excel('data/Synthetic Data set .xlsx')


X = np.column_stack((np.ones(len(data)), data.iloc[:, 1:3].values))
y = data.iloc[:, 0].values


initial_weights = np.array([1, 5, 1] , dtype='float64')

learning_rate = 0.004
rounds = 3000

trained_weights = perceptron_mse(X, y, learning_rate, rounds, initial_weights)


plt.scatter(X[:, 1], y, c=data.iloc[:, -1], cmap='RdYlGn')
plt.title('Actual vs. Predicted with Hyperplane')
plt.xlabel('Feature 1')
plt.ylabel('Actual')


x_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
hyperplane = trained_weights[0] + trained_weights[1] * x_range
plt.plot(x_range, hyperplane, color='black', linewidth=2, label='Hyperplane')
plt.legend()

plt.show()

print('Trained Weights:', trained_weights)
