# Hand-written logistic regression

import numpy as np
import pandas as pd

'''
Has no header
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
'''
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
# replace ? with Na and drop the rows with Na
df = pd.read_csv(data_url, sep = ',', header = None, index_col = 0).replace('?', np.nan).dropna()
df = df.astype('int64')
# convert benign to 0 and malignant to 1
df[10] = df[10].mask(df[10] == 2, 0)
df[10] = df[10].mask(df[10] == 4, 1)
df = np.array(df)

# add bias to the data frame
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
iterations = 100
theta = np.ones((n, 1))

# predict
def predict(x, theta):
	z = np.dot(x, theta)
	return 1 / (1 + np.exp(-z))

# train the logistic regression with stochastic gradient descent
for iteration in range(iterations):
	for i in range(m):
		h = predict(x_train[i], theta)
		# J = -1/m * sum over m [(y * log(h) + (1 - y) * log(1 - h))], where m is 1 here
		J = - (y_train[i] * np.log(h) + (1 - y_train[i]) * np.log(1 - h))

		print('theta:')
		print(theta)
		print('cost: ' + str(J))

		# theta -= alpha * sum over m [(h - y) * x], where m is 1 here
		theta -= alpha * (h - y_train[i]) * x_train[i].reshape((n, -1))

# model evaluation on the test data

pred = predict(x_test, theta)
pred = np.where(pred > 0.5, 1, 0)	# np.where(con, do, else_do)

# building confusion matrix

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(y_test)):
	if y_test[i] == 0:	# neg obs
		if pred[i] == 0:	# neg pred
			TN += 1
		else:	# pos pred
			FP += 1
	
	else:	# pos obs
		if pred[i] == 0:
			FN += 1
		else:
			TP += 1

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)