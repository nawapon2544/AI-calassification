import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

def update_weights(X, y, weights, learning_rate):
    y_pred = np.dot(X, weights)
    error = y - y_pred
    gradient = -2 * np.dot(X.T, error) / len(X)
    weights -= learning_rate * gradient
    return weights

def perceptron_mse(X, y, learning_rate, epochs, initial_weights=None):
    if initial_weights is None:

        initial_weights = np.array([1,5,1,3,4,5,1,2,3,4,8,2,8,4,1,2,3,5,9,1,5,3,1], dtype='float64')

    weights = initial_weights

    for epoch in range(epochs):
        weights = update_weights(X, y, weights, learning_rate)
        y_pred = np.dot(X, weights)
        mse = mean_squared_error(y, y_pred)
        r2 = r_squared(y, y_pred)
        print(f'Epoch {epoch+1}/{epochs}, MSE: {mse}, R-squared: {r2}')
    return weights


data = pd.read_excel('new/t1.xlsx')


X = np.column_stack((np.ones(len(data)), data.iloc[:, 1:].values))
y = data.iloc[:, 0].values


learning_rate = 0.001
epochs = 1000


initial_weights = np.array([1,5,1,3,4,5,1,2,3,4,8,2,8,4,1,2,3,5,9,1,5,3,4], dtype='float64')

trained_weights = perceptron_mse(X, y, learning_rate, epochs, initial_weights)


y_pred = np.dot(X, trained_weights)
plt.scatter(y, y_pred)
plt.title('Actual vs. Predicted with Hyperplane')
plt.xlabel('Actual')
plt.ylabel('Predicted')


x_range = np.linspace(min(y), max(y), 100)
hyperplane = trained_weights[0] + trained_weights[1] * x_range
plt.plot(x_range, hyperplane, color='red', linewidth=2, label='Hyperplane')
plt.legend()

plt.show()


print('Trained Weights:', trained_weights)
