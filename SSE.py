import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sum_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def update_weights(X, y, weights, learning_rate):
    y_pred = np.dot(X, weights)
    error = y - y_pred
    gradient = -2 * np.dot(X.T, error) / len(X)
    weights -= learning_rate * gradient
    return weights

def perceptron_sse(X, y, learning_rate, rounds, initial_weights=None):
    if initial_weights is None:
        weights = np.zeros(X.shape[1], dtype='float64')
    else:
        weights = initial_weights.astype('float64')

    y = y.astype('float64')

    for epoch in range(rounds):
        weights = update_weights(X, y, weights, learning_rate)
        y_pred = np.dot(X, weights)
        sse = sum_squared_error(y, y_pred)
        print(f'Epoch {epoch+1}/{rounds}, SSE: {sse}')
    return weights

# อ่านข้อมูลจาก Excel
data = pd.read_excel('data/Synthetic Data set .xlsx')

# แปลงข้อมูลเป็น NumPy arrays
X = np.column_stack((np.ones(len(data)), data.iloc[:, 1:].values))
y = data.iloc[:, 0].values

# กำหนด weights เริ่มต้น
initial_weights = np.array([1, 5, 1, 3], dtype='float64')

learning_rate = 0.005
rounds = 2500
# เทรน perceptron algorithm ด้วย SSE
trained_weights_sse = perceptron_sse(X, y, learning_rate, rounds, initial_weights)

# พล็อตกราฟ
plt.scatter(y, np.dot(X, trained_weights_sse), c=data.iloc[:, -1], cmap='RdYlGn')
plt.title('Actual vs. Predicted with Hyperplane (SSE)')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# เพิ่มเส้น hyperplane
x_range_sse = np.linspace(min(y), max(y), 100)
hyperplane_sse = trained_weights_sse[0] + trained_weights_sse[1] * x_range_sse
plt.plot(x_range_sse, hyperplane_sse, color='black', linewidth=2, label='Hyperplane (SSE)')
plt.legend()

plt.show()

# พิมพ์น้ำหนักที่เทรนได้
print('Trained Weights (SSE):', trained_weights_sse)
