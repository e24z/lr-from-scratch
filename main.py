# We will use pandas for data analysis/manipulation and matplotlib for data visualisation.
import pandas as pd
import matplotlib.pyplot as plt
# plt will not be necessary in final version i.e. a composable data analysis library

data = pd.read_csv('./sales-data.csv')

def mean_squared_error(beta_0, beta_1, points) -> float:
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].tv
        y = points.iloc[i].sales
        total_error += (y - (beta_1 * x + beta_0)) ** 2

    return total_error / float(len(points))

def gradient_descent(points, iterations, learning_rate):
    beta_0 = 0
    beta_1 = 0 
    n = float(len(points))

    for i in range(iterations):
        grad_beta_0 = 0
        grad_beta_1 = 0

        for j in range (len(points)):
            x = points.iloc[j].tv
            y = points.iloc[j].sales
            grad_beta_0 += -(2/n) * (y - (beta_1 * x + beta_0))
            grad_beta_1 += -(2/n) * x * (y - (beta_1 * x + beta_0))

        beta_0 = beta_0 - learning_rate * grad_beta_0
        beta_1 = beta_1 - learning_rate * grad_beta_1
    
    return (beta_0, beta_1)

beta_0, beta_1 = gradient_descent(data, 2000, 0.00001)
print('Hello, you should only be seeing this after the parameters')

plt.scatter(data.tv, data.sales)
y_pred = beta_1 * data.tv + beta_0
plt.plot(data.tv, y_pred, color='red', label='Regression Line')
plt.show()

