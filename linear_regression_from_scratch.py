# Here is a linear regression machine learning script that is pure hand written.

import numpy as np
import pandas as pd
import random

# load the data
exp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = np.array(pd.read_csv(exp_url, sep = ';', header = None))

# add bias to the dataframe
m_total = df.shape[0]
bias = np.ones((m_total, 1))
df = np.append(bias, df, axis = 1)

# split the data (80% training and 20% test)
np.random.shuffle(df)
# m = number of training sample, n = number of features (include the bias)
m = int(0.8 * m_total)
n = df.shape[1] - 1

x_train = df[:m, :-1]
# y is a flatten array, so needs to be transformed into an m*1 vector
y_train = df[:m, -1].reshape((m, 1))

x_test = df[m:, :-1]
y_test = df[m:, -1].reshape((m_total - m, 1))

# set up parameters
alpha = 0.0005	# learning rate
iterations = 10000
theta = np.ones((n, 1))

# training
for iteration in range(iterations):
	print('iteration = ' + str(iteration + 1))
	print('theta = ' + str(theta))

	# prediction
	h = np.dot(x_train, theta)
	# cost
	J = (((h - y_train) ** 2).sum(axis = 0)) / (2 * m)
	# update theta using batch gradient descent
	theta -= (alpha / m) * np.dot(x_train.T, (h - y_train))

	print('cost = ' + str(J))




