import numpy as np
import matplotlib.pyplot as plt
from numpy import random

# Assuming your data is tab-separated, adjust the delimiter accordingly
dataset = np.loadtxt("data/Synthetic Data set .txt", delimiter='\t')

weight = np.random.choice(np.arange(-2, 2), 3)  # random weight
weight0 = weight
learning_rate = 0.02
amax = 0.0

for j in range(len(dataset)):
    x = dataset[j][0]
    if j == 0:
        amin = x
        continue
    if x < amin:
        amin = x
    elif x > amax:
        amax = x

result_x = []
set_y = []
cal_x = np.array([0, 0, 0])
ck = []
r = 0
result = 0  # Initialize result here

while True:
    for i in range(len(dataset)):
        x = dataset[i]
        data = np.array(x[0:2])
        if int(x[2]) == 0:
            result = -1 * ((weight[0] * data[0]) + (weight[1] * data[1]) + (weight[2] * 1))
        elif int(x[2]) == 1:
            result = 1 * ((weight[0] * data[0]) + (weight[1] * data[1]) + (weight[2] * 1))

        if result >= 0:
            set_y.append(i)
            result_x.append(result)

    if r == 0:
        ck.insert(0, set_y[0])
    elif len(set_y) < ck[0]:
        ck[0] = len(set_y)

    print('round ', r)
    print('weight =', weight)
    Wr0 = (weight0)  # weight start
    print('wr0 =', Wr0)
    print('------------------------------------')
    print('------------------------------------')

    for j in range(len(set_y)):
        # update weight
        x = dataset[set_y[j]]
        data_1 = np.array([x[0], x[1], 1])

        if int(x[2]) == 0:
            cal_x = cal_x + (-1 * data_1)
        elif int(x[2]) == 1:
            cal_x = cal_x + (1 * data_1)

    if len(set_y) == 10 or r == 1000:  # r = Maximum iteration
        break
    else:
        weight = weight - (learning_rate * cal_x)
        result_x.clear()
        set_y.clear()
        cal_x = np.array([0, 0, 0])
        r += 1

plt_x1 = []
plt_y1 = []
plt_x2 = []
plt_y2 = []

for a in range(len(dataset)):
    if dataset[a][2] == 0:
        plt_x1.insert(a, dataset[a][0])
        plt_y1.insert(a, dataset[a][1])
    elif dataset[a][2] == 1:
        plt_x2.insert(a, dataset[a][0])
        plt_y2.insert(a, dataset[a][1])

plt01 = []
plt02 = []
plt03 = []
plt04 = []

for a in range(len(dataset)):
    if dataset[a][2] == 0:
        plt01.insert(a, dataset[a][0])
        plt02.insert(a, dataset[a][1])
    elif dataset[a][2] == 1:
        plt03.insert(a, dataset[a][0])
        plt04.insert(a, dataset[a][1])

print(plt01, plt02, plt03, plt04)
plt.scatter(plt01, plt02, color='salmon')
plt.scatter(plt03, plt04, color='plum')

x = [amin, amax]
y = []

for i in range(len(x)):
    n = (((weight[0] * -1) * x[i]) + (weight[2] * -1)) / weight[1]
    print(x[i], n)
    y.insert(i, n)

plt.plot([x[0], y[0]], [x[1], y[1]], 'black')
plt.show()
