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
    return weights.astype('float64')

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
data = pd.read_excel('data/test2.xlsx')

# แปลงข้อมูลเป็น NumPy arrays
X = np.column_stack((np.ones(len(data)), data.iloc[:, 1:].values))
y = data.iloc[:, 0].values

# กำหนด weights เริ่มต้น
initial_weights = np.array([1, 5, 1, 3] , dtype='float64')

learning_rate = 0.25
rounds = 5

# เทรน perceptron algorithm
trained_weights = perceptron_sse(X, y, learning_rate, rounds, initial_weights)

# พล็อตกราฟ
plt.scatter(y, np.dot(X, trained_weights), c=data.iloc[:, -1], cmap='RdYlGn')
plt.title('Actual vs. Predicted with Hyperplane')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# เพิ่มเส้น hyperplane
x_range = np.linspace(min(y), max(y), 100)
hyperplane = trained_weights[2] + trained_weights[1] * x_range
plt.plot(x_range, hyperplane, color='black', linewidth=2, label='Hyperplane')
plt.legend()

# หาความแม่นยำของเส้น
y_pred = np.dot(X, trained_weights)
sse = sum_squared_error(y, y_pred)
accuracy = 1 - sse / np.sum((y - np.mean(y))**2)
print(f'SSE: {sse:.4f}')
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.show()

# พิมพ์น้ำหนักที่เทรนได้
print('Trained Weights:', trained_weights)
