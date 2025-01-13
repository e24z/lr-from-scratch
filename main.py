import pandas as pd
import numpy as np
import matplotlib as plt

data = pd.read_csv('./sales-data.csv')
print(data)

# We assume that this data set can be modelled linearly.

# Function to get prediction given a model and an x value.