# We will use pandas for data analysis/manipulation and matplotlib for data visualisation.
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./sales-data.csv')

def mean_squared_error(m, b, points):
    """
        Description:
            The loss function. This will quantify the error between my model and the actual values in sales_data.csv.
        Parameters:
            m: the gradient
            b: the intercept
            points: the actual values from the sales dataset.
        Returns:
            float: the mean squared error.
    """
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].tv
        y = points.iloc[i].sales
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))
