# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # independent variables
y = dataset.iloc[:, -1].values # dependent variables


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

"""
Difference in scale between variables can cause issues in ML modelling 

Difference between two points calculated by Euclidian distance, larger values will dominate such calculation 
Therefore scaling is necessary!
"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# for training set, first fitting then transform
X_train = sc.fit_transform(X_train)
# for test set, only transform (no fitting)
X_test = sc.transform(X_test)