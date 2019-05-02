from sklearn.datasets import load_iris

iris = load_iris()

#print(iris)

print(iris.data.shape)
print(iris.target.shape)


X = iris.data
y = iris.target


from sklearn.neighbors import KNeighborsClassifier

# model is the estimator (instantiate the estimator)

knn = KNeighborsClassifier(n_neighbors=1)

# fit model with data (model learns relationship between X and y)

knn.fit(X,y)

# predict response for new observation

X_new = [[3,5,4,2], [5,4,3,2]]

print(knn.predict(X_new))


# Same step-procedure can also be used for other models such as logistic regression

from sklearn.linear_model import LogisticRegression

#logreg = LogisticRegression()

#logreg.fit(X,y)

#print(logreg.predict(X_new))

# Model evaluation (Evaluation process - Train and test dataset)

# using the actual iris dataset
#y_pred = logreg.predict(X)

#print(len(y_pred))

from sklearn import metrics

# training accuracy
#print(metrics.accuracy_score(y, y_pred))

from sklearn.metrics import classification_report, confusion_matrix

#print(confusion_matrix(y, y_pred))
#print(classification_report(y,y_pred))

# considered problematic zu train and test on the same data!

# to avoid this the train-test-split method is applied

# split dataset into two pieces: training and testing
# train model on training set
# test model on testing set

from sklearn.model_selection import train_test_split

# X train and test hold the pedal measurements (split in two sets), while y holds the flower category (split in two)

# test_size determines how much percent of data will be assigned to testing set (rest goes to training set)
# random state makes sure that the data is always split the same way!
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=4)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
import numpy as np

# "Elbow method"
# Error Rate vs K Value (checking which value for n_neighbors yields the lowest error rate)
"""
error_rate = []


for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.show()
"""

# alternative method: plotting testing accuracy

k_range = range(1,26)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.show()
